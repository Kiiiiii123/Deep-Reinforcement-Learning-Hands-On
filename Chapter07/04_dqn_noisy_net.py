import gym
import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tensorboardX import SummaryWriter

from lib import dqn_model, common


# 重新定义DQN网络
class NoisyDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(NoisyDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        # 线性层加入权重噪声
        self.noisy_layers = [
            model.NoisyLinear(conv_out_size, 512),
            model.NoisyLinear(512, n_actions)
        ]
        self.fc = nn.Sequential(
            # 列表中的第一个
            self.noisy_layers[0],
            nn.ReLU(),
            # 列表中的第二个
            self.noisy_layers[1]
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        fx = fx.type(torch.FloatTensor).cuda()
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.fc(conv_out)

    def noisy_layers_sigma_snr(self):
        return
