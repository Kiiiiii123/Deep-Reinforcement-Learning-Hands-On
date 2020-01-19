import gym
import ptan
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 4
NUM_ENVS = 15

# BATCH_SIZE用下面两个超参数进行替代
# 定义每个子进程进行loss和gradient计算的batch大小
GRAD_BATCH = 64
# 在单个GPU上进行测试时的选择2，是每次中心进程进行SGD迭代时从子进程获取的gradients batch
TRAIN_BATCH = 2
# 每个优化步骤需要GRAD_BATCH * TRAIN_BATCH个样本

ENV_NAME = 'PongNoFrameskip-v4'
NAME = 'pong'
REWARD_BOUND = 18


