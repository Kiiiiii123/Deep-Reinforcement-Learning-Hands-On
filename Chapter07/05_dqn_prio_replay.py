import gym
import ptan
import numpy as np
import argparse

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common

# 初始优先度设置
PRIO_REPLAY_ALPHA = 0.6
# β在100000帧后从0.4增长到1.0
BETA_START = 0.4
BETA_FRAMES = 100000

class PrioReplayBuffer:
    def __init__(self, exp_spurce, buf_size, prob_alpha=0.6):
        # 经验样本迭代器
        self.exp_source_iter = iter(exp_spurce)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        # numpy数组保存优先度
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

        def __len__(self):
            return len(self.buffer)

        # 存储样本进Ｂbuffer时便需要考虑优先度
        def populate(self, count):
            max_prio = self.priorities.max() if self.buffer else 1.0
            for _ in range(count):
                sample = next(self.exp_source_iter)
                # 未满则继续加入
                if len(self.buffer) < self.capacity:
                    self.buffer.append(sample)
                # 已满则进行样本替换
                else:
                    self.buffer[self.pos] = sample
                self.priorities[self.pos] = max_prio
                self.pos = (self.pos + 1) % self.capacity

        # 采样时将优先度转换成采样概率
        def sample(self, batch_size, beta=0.4):
            if len(self.buffer) == self.capacity:
                prios = self.priorities
            else:
                prios = self.priorities[:self.pos]
            probs = prios ** self.prob_alpha
            probs /= probs.sum()






