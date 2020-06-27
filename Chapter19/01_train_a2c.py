import os
import time
import ptan
import gym
import roboschool
import argparse
from tensorboardX import SummaryWriter

from lib import model, test_net

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = 'RoboschoolHalfCheetah-v1'
ENVS_COUNT = 16
GAMMA = 0.99
STEPS_COUNT = 5
TEST_ITERS = 100000
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-3


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

    opt_act = optim.Adam(model_act.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(model_crt.parameters(), lr=LEARNING_RATE_CRITIC)

    exp_batch = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for step_idx, exp in exp_source:
                rewards_steps = exp_source.pop_rewards_steps()
                if rewards_steps:
                    rewards, steps = zip(*rewards_steps)
                    tb_tracker.track('episode_steps', np.mean(steps), step_idx)
                    tracker.reward(np.mean(rewards), step_idx)

                if step_idx % TEST_ITERS == 0:
                    ts = time.time()
                    rewards, steps = test_net(model_act, test_env, device=device)

