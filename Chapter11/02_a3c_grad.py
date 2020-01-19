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


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))


# 子进程中执行的函数，每个子进程需要执行的操作很多，从搜集数据样本增加到自己计算loss和梯度，有自己的Tensorboard监测数据，train_dequeue将计算的梯度传递给中心进程
def grads_func(proc_name, net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]

    agent = ptan.agent.PolicyAgent(lambda x:net(x)[0], device=device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    batch = []
    frame_idx = 0
    writer = SummaryWriter(comment=proc_name)

    with common.RewardTracker(writer, stop_reward=REWARD_BOUND) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for exp in exp_source:
                frame_idx += 1
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards and tracker.reward(new_rewards[0], frame_idx):
                    break
                batch.append(exp)
                if len(batch) < GRAD_BATCH:
                    continue




