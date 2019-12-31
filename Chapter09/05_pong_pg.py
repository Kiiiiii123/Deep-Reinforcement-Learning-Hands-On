import gym
import ptan
import numpy as np
import argparse
import collections
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import torch.optim as optim

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.0001
# 用于计算熵损失
ENTROPY_BETA = 0.1
BATCH_SIZE = 128

REWARD_STEPS = 10
# 最近一万帧求baseline
BASELINE_STEPS = 1000000
# 梯度裁切
GRAD_L2_CLIP = 0.1
# 并行采样环境数量
ENV_COUNT = 32


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make("PongNoFrameskip-v4"))


# 用于加速计算移动平均的baseline，队列数据结构
class MeanBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.deque = collections.deque(maxlen=capacity)
        self.sum = 0.0

    def add(self, val):
        if len(self.deque) == self.capacity:
            self.sum -= self.deque[0]
        self.deque.append(val)
        self.sum += val

    def mean(self):
        if not self.deque:
            return 0.0
        return self.sum / len(self.deque)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("-n", "--name", required=True, help="Name of the run")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # 创建环境组合
    envs = [make_env() for _ in range(ENV_COUNT)]
    writer = SummaryWriter(comment="-pong-pg-" + args.name)

    net = common.AtariPGN(envs[0].observation_space.shape, envs[0].action_space.n).to(device)
    print(net)

    agent = ptan.agent.PolicyAgent(net, apply_softmax=True, device=device)
    # 多并行环境进行采样
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    total_rewards = []
    step_idx = 0
    done_episodes = 0
    baseline_buf = MeanBuffer(BASELINE_STEPS)

    batch_states, batch_actions, batch_scales = [], [], []
    sum_reward = 0.0

    with common.RewardTracker(writer, stop_reward=18) as tracker:
        for step_idx, exp in enumerate(exp_source):
            baseline_buf.add(exp.reward)
            # 近100万个样本的均值构成baseline
            baseline = baseline_buf.mean()
            batch_states.append(np.array(exp.state, copy=False))
            batch_actions.append(int(exp.action))
            batch_scales.append(exp.reward - baseline)

            # handle new rewards
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if tracker.reward(new_rewards[0], step_idx):
                    break

            if len(batch_states) < BATCH_SIZE:
                continue

            states_v = torch.FloatTensor(batch_states).to(device)
            batch_actions_t = torch.LongTensor(batch_actions).to(device)

            # 计算矩阵全局标准差
            scale_std = np.std(batch_scales)
            batch_scale_v = torch.FloatTensor(batch_scales).to(device)

            optimizer.zero_grad()
            logits_v = net(states_v)
            log_prob_v = F.log_softmax(logits_v, dim=1)
            log_prob_actions_v = batch_scale_v * log_prob_v[range(BATCH_SIZE), batch_actions_t]
            loss_policy_v = -log_prob_actions_v.mean()

            prob_v = F.softmax(batch_states, dim=1)
            entropy_v = -(prob_v * log_prob_v).sum(dim=1).mean()
            entropy_loss_v = -ENTROPY_BETA * entropy_v
            loss_v = loss_policy_v + entropy_loss_v
            loss_v.backward()
            # 梯度裁切
            nn_utils.clip_grad_norm_(net.parameters(), GRAD_L2_CLIP)
            optimizer.step()

            # 计算KL散度
            new_logits_v = net(states_v)
            new_prob_v = F.softmax(new_logits_v, dim=1)
            kl_div_v = -((new_prob_v / prob_v).log() * prob_v).sum(dim=1).mean()
            writer.add_scalar("kl", kl_div_v.item(), step_idx)

            grad_max = 0.0
            grad_means = 0.0
            grad_count = 0
            for p in net.parameters():
                m_grad_max = max(m_grad_max, p.grad.abs().max().item())
                grad_means += (p.grad ** 2).mean().sqrt().item()
                grad_count += 1

            writer.add_scalar("baseline", baseline, step_idx)
            writer.add_scalar("entropy", entropy_v.item(), step_idx)
            writer.add_scalar("batch_scales", np.mean(batch_scales), step_idx)
            writer.add_scalar("batch_scales_std", scale_std, step_idx)
            writer.add_scalar("loss_entropy", entropy_loss_v.item(), step_idx)
            writer.add_scalar("loss_policy", loss_policy_v.item(), step_idx)
            writer.add_scalar("loss_total", loss_v.item(), step_idx)
            writer.add_scalar("grad_l2", grad_means / grad_count, step_idx)
            writer.add_scalar("grad_max", grad_max, step_idx)

            batch_states.clear()
            batch_actions.clear()
            batch_scales.clear()

        writer.close()







