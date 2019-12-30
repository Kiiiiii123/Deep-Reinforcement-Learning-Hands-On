import gym
import ptan
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


GAMMA = 0.99
LEARNING_RATE = 0.001
# scale of entropy bonus
ENTROPY_BETA = 0.01
BATCH_SIZE = 8
# 贝尔曼方程向前铺开多少步来估计每个转换的折扣奖励
REWARD_STEPS = 10


# vanilla PG method
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

