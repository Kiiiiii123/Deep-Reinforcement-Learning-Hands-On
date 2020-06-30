import os
import time
import ptan
import gym
import math
import roboschool
import argparse
from tensorboardX import SummaryWriter

from lib import model, test_net, calc_logprob

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


ENV_ID = 'RoboschoolHalfCheetah-v1'
GAMMA = 0.99
GAE_LAMBDA = 0.95

TRAJECTORY_SIZE = 2049
LEARNING_RATE_ACTOR = 1e-5
LEARNING_RATE_CRITIC = 1e-4

PPO_EPS = 0.2
PPO_EPOCHES = 10
PPO_BATCH_SIZE = 64

TEST_ITERS = 100000


def calc_adv_ref(trajectory, net_crt, states_v, device='cpu'):
    """
    Takes the trajectory with steps and calculates advantages for actor and reference values for critic training.
    :param trajectory: trajectory list
    :param net_crt: critic network
    :param states_v: states tensor
    :return: tuple with advantage numpy array and reference values
    """
    values_v = net_crt(states_v)
    values = values_v.squeeze().data.cpu().numpy()
    # generalized advantage estimator: smoothed version of the advantage
    last_gae = 0.0
    result_adv = []
    result_ref = []
    for val, next_val, (exp,) in zip(reversed(values[:-1]),
                                     reversed(values[1:]),
                                     reversed(trajectory[:-1])):
        if exp.done:
            delta = exp.reward - val
            last_gae = delta
        else:
            delta = exp.reward + next_val - val
            last_gae = delta + GAMMA * GAE_LAMBDA * last_gae
        result_adv.append(last_gae)
        result_ref.append(last_gae + val)

    adv_v = torch.FloatTensor(list(reversed(result_adv))).to(device)
    ref_v = torch.FloatTensor(list(reversed(result_ref))).to(device)
    return adv_v, ref_v

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Enable CUDA')
    parser.add_argument('-n', '--name', required=True, help='Name of the run')
    parser.add_argument('-e', '--env', default=ENV_ID, help='Environment id, default=' + ENV_ID)
    parser.add_argument('--lrc', default=LEARNING_RATE_CRITIC, type=float, help='Critic learning rate')
    parser.add_argument('--lra', default=LEARNING_RATE_ACTOR, type=float, help='Actor learning rate')
    args = parser.parse_args()
    device = torch.device('cuda' if args.cuda else 'cpu')

    save_path = os.path.join('saves', 'ppo-' + args.name)
    os.makedirs(save_path, exist_ok=True)

    env = gym.make(args.env)
    test_env = gym.make(args.env)

    net_act = model.ModelActor(env.observation_space.shape[0], env.action_space.n).to(device)
    net_crt = model.ModelCritic(env.observation_space.shape[0]).to(device)
    print(net_act)
    print(net_crt)

    writer = SummaryWriter(comment='-ppo_' + args.name)
    agent = model.AgentA2C(net_act, device=device)
    exp_source = ptan.experience.ExperienceSource(env, agent, steps_count=1)

    opt_act = optim.Adam(net_act.parameters(), lr=args.lra)
    opt_crt = optim.Adam(net_crt.parameters(), lr=args.lrc)

    trajectory = []
    best_reward = None
    with ptan.common.utils.RewardTracker(writer) as tracker:
        for step_idx, exp in enumerate(exp_source):
            rewards_steps = exp_source.pop_rewards_steps()
            if rewards_steps:
                rewards, steps = zip(*rewards_steps)
                writer.add_scalar('episode_steps', np.mean(steps), step_idx)
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

            trajectory.append(exp)
            if len(trajectory) < TRAJECTORY_SIZE:
                continue

            traj_states = [t[0].state for t in trajectory]
            traj_states_v = torch.FloatTensor(traj_states).to(device)
            traj_actions = [t[0].action for t in trajectory]
            traj_actions_v = torch.FloatTensor(traj_actions).to(device)
            traj_adv_v, traj_ref_v = calc_adv_ref(trajectory, net_crt, traj_states_v, device=device)








            #states_v, actions_v, vals_ref_v = common.unpack_bacth_a2c(exp_batch, net_crt, GAMMA ** STEPS_COUNT, device)
            #exp_batch.clear()

            #opt_crt.zero_grad()
            #value_v = net_crt(states_v)
            #loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)
            #loss_value_v.backward()
            #opt_crt.step()

            #opt_act.zero_grad()
            #mu_v = net_act(states_v)
            #adv_v = vals_ref_v.unsqueeze(dim=-1) - value_v.detach()
            #log_prob_v = adv_v * calc_logprob(mu_v, net_act.logstd, actions_v)
            #loss_policy_v = -log_prob_v.mean()
            #entropy_loss_v = ENTROPY_BETA * (-(torch.log(2*math.pi*torch.exp(net_act.logstd)) + 1)/2).mean()
            #loss_v = loss_policy_v + entropy_loss_v
            #loss_v.backward()
            #opt_act.step()


