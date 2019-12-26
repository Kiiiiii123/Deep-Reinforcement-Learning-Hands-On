import gym
import ptan
import numpy as np
import argparse

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common

# 初始优先度设置
PRIO_REPLAY_ALPHA = 0.6
# β在100000帧后从0.4增长到1.0
BETA_START = 0.4
BETA_FRAMES = 100000

class PrioReplayBuffer:
    def __init__(self, exp_spurce, buf_size, prob_alpha=0.6):
        # 经验样本迭代器
        self.exp_source_iter = iter(exp_spurce)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        # numpy数组保存优先度
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

        def __len__(self):
            return len(self.buffer)

        # 存储样本进buffer时便需要考虑优先度
        def populate(self, count):
            max_prio = self.priorities.max() if self.buffer else 1.0
            for _ in range(count):
                sample = next(self.exp_source_iter)
                # 未满则继续加入
                if len(self.buffer) < self.capacity:
                    self.buffer.append(sample)
                # 已满则进行样本替换
                else:
                    self.buffer[self.pos] = sample
                # 新加入的样本优先度最高
                self.priorities[self.pos] = max_prio
                # 样本的加入按照顺序
                self.pos = (self.pos + 1) % self.capacity

        # 采样时将优先度转换成采样概率
        def sample(self, batch_size, beta=0.4):
            # 根据优先度计算被概率，根据概率进行采样
            if len(self.buffer) == self.capacity:
                prios = self.priorities
            else:
                prios = self.priorities[:self.pos]
            # 优先度高的变得更重要
            probs = prios ** self.prob_alpha
            probs /= probs.sum()
            # 采样
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            samples = [self.buffer[idx] for idx in indices]
            # 为采样出的样本设置权重用于补偿独立同分布
            total = len(self.buffer)
            weights = (total * probs[indices]) ** (-beta)
            weights /= weights.max()
            # 索引用于为被采样的样本更新优先度
            return samples, indices, weights

        # 根据计算获得的loss更新优先度
        def update_priorities(self, batch_indices, batch_priorities):
            for idx, prio in zip(batch_indices, batch_priorities):
                self.priorities[idx] = prio

        # 借助权重计算loss
        def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
            states, actions, rewards, dones, next_states = common.unpack_batch(batch)

            states_v = torch.tensor(states).to(device)
            next_states_v = torch.tensor(next_states).to(device)
            actions_v =torch.tensor(actions).to(device)
            rewards_v = torch.tensor(rewards).to(device)
            done_mask = torch.ByteTensor(dones).to(device)
            batch_weights_v = torch.tensor(batch_weights).to(device)

            state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
            next_state_values = tgt_net(next_states_v).max(1)[0]
            next_state_values[done_mask] = 0.0

            expected_state_action_values = next_state_values.detach() * gamma +rewards_v
            # 自实现的考虑到样本权重的损失函数
            losses_v = batch_weights_v * (state_action_values - expected_state_action_values) ** 2
            # 防止优先度为0的情况发生
            return losses_v.mean(), losses_v + 1e-5


if __name__ == "__main__":
    params = common.HYPERPARAMS['pong']
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-prio-replay")
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    # 使用优先度的re[lay buffer而不是普通的
    buffer = PrioReplayBuffer(exp_source, params['replay_size'], PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    # 准备训练循环
    frame_idx = 0
    beta = BETA_START

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)
            # 递增的β策略
            beta = min(1.0, BETA_START + frame_idx * (1.0 - BETA_START) / BETA_FRAMES)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                writer.add_scalar("beta", beta, frame_idx)
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break
            if len(buffer) < params['replay_init']:
                continue
            optimizer.zero_grad()
            # 返回三样东西
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size', beta])
            loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net.target_model, params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()
            # 更新样本权重
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()




