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


# 以完整episode的奖励为输入，以截止某一个step时的奖励列表
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

    # 使用全新的agent接口，对输出使用softmax将输出转换为概率
    agent = ptan.agent.PolicyAgent(net, preprocessor=ptan.agent.float32_preprocessor, apply_softmax=True)
