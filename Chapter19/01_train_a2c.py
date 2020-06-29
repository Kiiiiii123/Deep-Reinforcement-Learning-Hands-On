import os
import time
import ptan
import gym
import math
import roboschool
import argparse
from tensorboardX import SummaryWriter

from lib import model, test_net, common, calc_logprob

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
BATCH_SIZE = 32
ENTROPY_BETA = 1e-3


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

    net_act = model.ModelActor(envs[0].observation_space.shape[0], envs[0].action_space.n).to(device)
    net_crt = model.ModelCritic(envs[0].observation_space.shape[0]).to(device)
    print(net_act)
    print(net_crt)

    writer = SummaryWriter(comment='-a2c_' + args.name)
    agent = model.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, GAMMA, STEPS_COUNT)

    opt_act = optim.Adam(net_act.parameters(), lr=LEARNING_RATE_ACTOR)
    opt_crt = optim.Adam(net_crt.parameters(), lr=LEARNING_RATE_CRITIC)

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
                    rewards, steps = test_net(net_act, test_env, device=device)
                    print('Test done in %.2f sec, reward %.3f, steps %d' % (time.time() - ts, rewards, steps))
                    writer.add_scalar('test_reward', rewards, step_idx)
                    writer.add_scalar('test_steps', steps, step_idx)
                    if best_reward is None or best_reward < rewards:
                        if best_reward is not None:
                            print('Best reward updated: %.3f -> %.3f' % (best_reward, rewards))
                            name = 'best_%.3f_%d.bat' % (rewards, step_idx)
                            fname = os.path.join(save_path, name)
                            torch.save(net_act.state_dict(), fname)
                        best_reward = rewards

                exp_batch.append(exp)
                if len(exp_batch) < BATCH_SIZE:
                    continue

                states_v, actions_v, vals_ref_v = common.unpack_bacth_a2c(exp_batch, net_crt, GAMMA ** STEPS_COUNT, device)
                exp_batch.clear()

                opt_crt.zero_grad()
                value_v = net_crt(states_v)
                loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
                loss_value_v.backward()
                opt_crt.step()

                opt_act.zero_grad()
                mu_v = net_act(states_v)
                adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
                log_prob_v = adv_v * calc_logprob(mu_v, net_act.logstd, actions_v)
                loss_policy_v = -log_prob_v.mean()
                entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*torch.exp(net_act.logstd)) + 1)/2).mean()
                loss_v = loss_policy_v + entropy_loss_v
                loss_v.backward()
                opt_act.step()

                tb_tracker.track('advantage', adv_v, step_idx)
                tb_tracker.track('values', value_v, step_idx)
                tb_tracker.track('batch_rewards', vals_ref_v, step_idx)
                tb_tracker.track('loss_entropy', entropy_loss_v, step_idx)
                tb_tracker.track('loss_policy', loss_policy_v, step_idx)
                tb_tracker.track('loss_value', loss_value_v, step_idx)
                tb_tracker.track('loss_total', loss_v, step_idx)

