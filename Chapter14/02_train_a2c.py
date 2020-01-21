import os
import time
import math
import ptan
import gym
import pybullet_envs
import argparse
from tensorboardX import SummaryWriter

from lib import model, common

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

ENV_ID = "MinitaurBulletEnv-v0"
GAMMA = 0.99
REWARD_STEPS = 2
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
ENTROPY_BETA = 1e-4

TEST_ITERS = 1000


# 在测试环境下对网络模型进行测试，不进行随机采样
def test_net(net, env, count=10, device='cpu'):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = ptan.agent.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)[0]
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


