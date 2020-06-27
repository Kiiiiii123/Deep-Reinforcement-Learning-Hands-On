import os
import ptan
import gym
import roboschool
import argparse
from tensorboardX import SummaryWriter

from lib import model

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


ENV_ID = 'RoboschoolHalfCheetah-v1'
ENVS_COUNT = 16
GAMMA = 0.99
STEPS_COUNT = 5


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    parser.add_argument('-n', '--name', required=True, help='Name of the run')
    parser.add_argument('-e', '--env', default=ENV_ID, help='Environment id, default=' + ENV_ID)
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    save_path = os.path.join('saves', 'a2c-' + args.name)
    os.makedirs(save_path, exist_ok=True)

    envs = [gym.make(args.env) for _ in range(ENVS_COUNT)]
    test_env = gym.make(args.env)

    model_act = model.ModelActor(envs[0].observation_space.shape[0], envs[0].action_space.n).to(device)
    model_crt = model.ModelCritic(envs[0].observation_space.shape[0]).to(device)
    print(model_act)
    print(model_crt)

    writer = SummaryWriter(comment='-a2c' + args.name)
    agent = model.AgentA2C(model_act, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, STEPS_COUNT)

    exp_batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:

