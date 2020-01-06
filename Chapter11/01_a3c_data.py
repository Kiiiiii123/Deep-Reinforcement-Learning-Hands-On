import gym
import ptan
import numpy as np

import argparse
import collections
from tensorboardX import SummaryWriter

import torch.nn.utils as nn_utils
import torch.functional as F
import torch.optim as optim
# Pytorch对多进程的封装
import torch.multiprocessing as mp

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 4
CLIP_GRAD = 0.1

# 预定进程数量
PROCESSES_COUNT = 4
NUM_ENVS = 15

ENV_NAME = "PongNoFrameskip-v4"
NAME = 'pong'
REWARD_BOUND = 18


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))


TotalReward = collections.namedtuple('TotalReward', field_names='reward')


# 在子进程中执行（与抓取、发送样本相关）
def data_func(net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device = device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_BOUND)





