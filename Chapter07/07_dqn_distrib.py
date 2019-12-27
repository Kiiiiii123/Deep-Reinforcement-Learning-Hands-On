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
            # 全连接层输出部分变为包含n_actions*N_ATOMS个元素的矩阵，包含每个动作额概率分布
            nn.Linear(512, n_actions * N_ATOMS)
        )

        # 注册不应被视为模型参数的缓冲区，_buffer中的元素不会被优化器更新，如果在模型中需要需要一些参数，并且要通过state_dict返回，且不需要被优化器训练，那么这些参数可以注册在_buffer中
        # torch.arange函数不包含末尾的按照指定间隔生成列表
        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        # 对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1，dim=1表示按行计算
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        fx = fx.type(torch.FloatTensor).cuda()
        conv_out = self.conv(fx).view(batch_size, -1)
        fc_out = self.fc(conv_out)
        return fc_out.view(batch_size, -1, N_ATOMS)

    # 对输出应用softmax函数并保持张量形式
    def apply_softmax(self, t):
        # 只针对单个分布应用softmax，因为求的是(state,action)对的奖励值，view函数起到的是reshape的作用
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())

    # 从概率分布中获取Q-values，奖励期望值
    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        # 按列相加求和
        res = weights.sum(dim=2)
        return cat_out, res

    # 提取Q-values
    def qvals(self, x):
        return self.both(x)[1]


def calc_loss(batch, net, tgt_net, gamma, device="cpu", save_prefix=None):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)

    # next state distribution
    next_distr_v, next_qvals_v = tgt_net.both(next_states_v)
    # 贪心策略选择动作用于与DQN形成对照组
    next_actions = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distr = tgt_net.apply_softmax(next_distr_v).data.cpu().numpy()

    # 最佳动作下的概率分布
    next_best_distr =next_distr[range(batch_size), next_actions]
    dones = dones.astype(np.bool)

    # 对概率分布进行投影，投影出的概率分布是期望网络输出的目标概率分布
    proj_distr = common.distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    # 求网路输出并计算两个分布之间的KL散度
    distr_v = net(states_v)
    # log_softmax函数在softmax函数的基础上在进行log操作
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)
    loss_v = -state_log_sm_v * proj_distr_v
    return loss_v.sum(dim=1).mean()






