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




