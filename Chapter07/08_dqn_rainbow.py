import gym
import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common

# n-step
REWARD_STEPS = 2

# priority replay
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

# C51
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self. _get_conv_out(input_shape)
        # 预测输入状态的值分布
        self.fc_val = nn.Sequential(
            dqn_model.NoisyLinear(conv_out_size, 256),
            nn.ReLU(),
            dqn_model.NoisyLinear(256, N_ATOMS)
        )

        # 预测游戏中每个动作的分布
        self.fc_adv = nn.Sequential(
            dqn_model.NoisyLinear(conv_out_size, 256),
            nn.ReLU(),
            dqn_model.NoisyLinear(256, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
         o = self.conv(torch.zeros(1, *shape))
         return int(np.prod(o.size()))

    # 获取动作值分布
    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        fx = fx.type(torch.FloatTensor).cuda()
        conv_out = self.conv(fx).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, -1, N_ATOMS)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + adv_out - adv_mean

    # 对输出的概率分布进行softmax
    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS).view(t.size()))

    def both(self, x):
        x = x.type(torch.FloatTensor).cuda()
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]


def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = common.un_pack(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    # 加入权重以进行优先级采样
    batch_weights_v = torch.tensor(batch_weights).to(device)

    # Double DQN方法，主网络选择动作，目标网络获取值



