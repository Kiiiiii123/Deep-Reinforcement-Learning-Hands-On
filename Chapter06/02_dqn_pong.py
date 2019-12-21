from lib import wrappers
from lib import dqn_model

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

