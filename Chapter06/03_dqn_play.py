import gym
import time
import argparse
import numpy as np
import torch

from lib import wrappers
from lib import dqn_model

import collections

DEFAULT_ENV_NAME = 'PongNoFrameskip-v4'
FPS = 25


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='MOdel file to load')
    parser.add_argument('-e', '--env', default=DEFAULT_ENV_NAME, help='Environment name to use, default='
                                                                      + DEFAULT_ENV_NAME)
    parser.add_argument('-r', '--record', help='Directory fot video')
    parser.add_argument('--no-vis', default=True, dest='vis', help='Disable visualization', action='store_false')
    args = parser.parse_args()

    env = wrappers.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)

    net = dqn_model.DQN(env.observation.shape, env.action_space.n)
    # The argument map_location is needed to map the loaded tensor location from GPU to CPU
    state = torch.load(args.model, map_location=lambda stg, _: stg)
    net.load_state_dict(state)

    state = env.reset()
    total_reward = 0.0
    c = collections.Counter()

