from Chapter06.lib import wrappers
from Chapter06.lib import dqn_model

import argparse
import time
import numpy as np
import collections

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
# 最后100个episodes平均分数达到条件才结束训练
MEAN_REWARD_BOUND = 19.5

# 用于贝尔曼近似
GAMMA = 0.99
# 从Replay Buffer采样的批次大小
BATCH_SIZE = 32
# Replay Buffer容量
REPLAY_SIZE = 10000
# Replay Buffer积累够了才开始训练
REPLAY_START_SIZE = 10000
# Adam最优化学习率
LEARNING_RATE = 1e-4
# 同步两个网络权重的频率
SYNC_TARGET_FRAMES = 1000

# epsilon数值变化策略
EPSILON_DECAY_LAST_FRAME = 10 ** 5
EPSILON_START = 1.0
EPSILON_FINAL = 0.02

Experience = collections.namedtuple('Experience', field_names=['State', 'action', 'reward', 'done', 'new_state'])


class ExperienceBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

        def __len__(self):
            return len(self.buffer)

        def append(self, experience):
            self.buffer.append(experience)

        # 从Buffer中按照batch大小进行样本数据的随机采样
        def sample(self, batch_size):
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            # 将对象中可迭代的元素打包成一个个元组，并返回元组构成的列表，*操作符的作用是将元组解包
            states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
            # 返回numpy数组方便后续计算loss
            return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(next_states)


# 智能体与环境交互并将结果存储到experience buffer中
class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    # epsilon-greedy策略
    def play_step(self, net, epsilon=0.0, device='cpu'):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward, is_done, new_state)
        self.exp_buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


# 传入两个网络才能计算loss
def calc_loss(batch, net, tgt_net, device='cpu'):
    states, actions, rewards, dones, next_states = batch
    # 将numpy数组转换成张量
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actions_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    # 创建boolen tensor
    done_mask = torch.ByteTensor(dones).to(device)

    # gather函数抓取选定动作后的Q-value
    state_action_value = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0
    next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA +rewards_v
    return nn.MSELoss()(state_action_value, expected_state_action_values)


if __name__ == "__main__":
    # 进行命令行解析
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="Name of the environment,default=" + DEFAULT_ENV_NAME)
    parser.add_argument("--reward", type=float, default=MEAN_REWARD_BOUND, help="Mean reward of boundary for stop of training, default=%.2f" % MEAN_REWARD_BOUND)
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrappers.make_env(args.env)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)

    writer = SummaryWriter(comment='-' + args.env)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_mean_reward = None

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)
        reward = agent.play_step(net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            mean_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games,mean reward %.3f, eps %.2f,speed %.2f f/s" % (frame_idx, len(total_rewards), mean_reward, epsilon, speed))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
            if best_mean_reward is None or best_mean_reward < mean_reward:
                # 保存模型
                torch.save(net.state_dict(), args.env + "-best.dat")
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f -> %.3f,model saved" % (best_mean_reward, mean_reward))
                    best_mean_reward = mean_reward
            if mean_reward > args.reward:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < REPLAY_START_SIZE:
            continue

        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device=device)
        loss_t.backward()
        optimizer.step()
    writer.close()







