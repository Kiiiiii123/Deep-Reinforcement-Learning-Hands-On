import gym
import ptan
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import common

SAVE_STATES_IMG = False
SAVE_TRANSITIONS_IMG = False

# 开启以保存概率分布简化Debug，训练过程可视化
if SAVE_STATES_IMG or SAVE_TRANSITIONS_IMG:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pylab as plt

# 设置值分布的范围，柱状体数量，计算之间的间距
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

# 拿出多少状态进行平均值计算以及多久更新一次
STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100


class DistributionalDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DistributionalDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8 ,stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)




