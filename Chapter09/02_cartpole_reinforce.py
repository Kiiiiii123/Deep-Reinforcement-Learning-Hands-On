import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

GAMMA = 0.99
LEARNING_RATE = 0.01
# 用于训练的完整episode数量
EPISODE_TO_TRAIN = 4


class PGN(nn.Module):
    def __init__(self, input_size, n_actions):
        super(PGN, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        return self.net(x)


# 以完整episode的奖励为输入，输出该步以后能得到的奖励
def calc_qvals(rewards):
    res = []
    sum_r = 0.0
    # 返回一个反转的迭代器
    for r in reversed(rewards):
        sum_r *= GAMMA
        sum_r += r
        res.append(sum_r)
    return list(reversed(res))


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    writer = SummaryWriter(comment="-cartpole-reinforce")

    net = PGN(env.observation_space.shape[0], env.action_space.n)
    print(net)

    # 使用全新的agent接口，对输出使用softmax将输出转换为概率，每观察一次输出一次动作概率，随机采样选择动作
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=GAMMA)

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # 记录整体情况
    total_rewards = []
    step_idx = 0
    done_episodes = 0

    # 抓取样本训练数据的变量
    batch_episodes = 0
    cur_rewards = []
    batch_states, batch_actions, batch_qvals = [], [], []

    # agent不断与环境交互采样
    for step_idx, exp in enumerate(exp_source):
        batch_states.append(exp.state)
        batch_actions.append(int(exp.action))
        cur_rewards.append(exp.reward)

        if exp.last_state is None:
            # local reward转换成Q-values，extend函数将可迭代对象中的每个元素逐个添加到列表中
            batch_qvals.extend(calc_qvals(cur_rewards))
            cur_rewards.clear()
            batch_episodes += 1

        # 在episode完结处执行，回报当前进展
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            done_episodes += 1
            reward = new_rewards[0]
            total_rewards.append(reward)
            # 取最近的100个奖励求平均
            mean_rewards = float(np.mean(total_rewards[-100:]))
            print("%d: reward: %6.2f, mean_100: %6.2f, episodes: %d"% (step_idx, reward, mean_rewards, done_episodes))
            writer.add_scalar("reward", reward, step_idx)
            writer.add_scalar("reward_100", mean_rewards, step_idx)
            writer.add_scalar("episodes", done_episodes, step_idx)
            if mean_rewards > 195:
                print("Solved in %d steps and %d episodes!"% (step_idx, done_episodes))
                break

        # 每次训练徐足够的完整episodes
        if batch_episodes < EPISODE_TO_TRAIN:
            # continue表示循环中后续的代码不再执行
            continue

        optimizer.zero_grad()
        states_v = torch.FloatTensor(batch_states)
        batch_actions_t = torch.LongTensor(batch_actions)
        batch_qvals_v = torch.FloatTensor(batch_qvals)

        # 计算loss，网络的分值输出为logits
        logits_v = net(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        # 由于Pytorch中的optimizer只进行损失函数的最小化，因此这里加负号
        loss_v = -log_prob_actions_v.mean()

        loss_v.backward()
        optimizer.step()

        # 每四个episodes训练一次并清零
        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

    writer.close()