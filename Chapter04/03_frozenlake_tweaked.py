import random
import gym
import gym.spaces
from collections import namedtuple
import numpy as np
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torch.optim as optim


HIDDEN_SIZE = 128
BATCH_SIZE = 100
PERCENTILE = 30
GAMMA = 0.9


class DiscreteOneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteOneHotWrapper, self).__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        shape = (env.observation_space.n, )
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape, dtype=np.float32)

    def observation(self, observation):
        res = np.copy(self.observation_space.low)
        res[observation] = 1.0
        return res


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x)


Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])


def iterate_batches(env, net, batch_size):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs = env.reset()
    sm = nn.Softmax(dim=1)
    while True:
        obs_v = torch.FloatTensor([obs])
        act_probs_v = sm(net(obs_v))
        act_probs = act_probs_v.data.numpy()[0]
        action = np.random.choice(len(act_probs), p=act_probs)
        next_obs, reward, is_done, _ = env.step(action)
        episode_reward += reward
        episode_step = EpisodeStep(observation=obs, action=action)
        episode_steps.append(episode_step)
        if is_done:
            episode = Episode(reward=episode_reward, steps=episode_steps)
            batch.append(episode)
            episode_reward = 0.0
            episode_steps = []
            next_obs = env.reset()
            if len(batch) == batch_size:
                yield batch
                batch = []
        obs = next_obs


def filter_batch(batch, percentile):
    disc_rewards = list(map(lambda episode: episode.reward * (GAMMA ** len(episode.steps)), batch))
    reward_bound = np.percentile(disc_rewards, percentile)

    train_obs = []
    train_act = []
    elite_batch = []
    for episode, disc_reward in zip(batch, disc_rewards):
        if disc_reward > reward_bound:
            train_obs.extend(map(lambda step: step.observation, episode.steps))
            train_act.extend(map(lambda step: step.action, episode.steps))
            elite_batch.append(episode)

    return elite_batch, train_obs, train_act, reward_bound


if __name__ == '__main__':
    random.seed(12345)
    env = DiscreteOneHotWrapper(gym.make('FrozenLake-v0'))
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    writer = SummaryWriter(comment='-frozenlake-tweaked')

    full_batch = []
    for iter_num, batch in enumerate(iterate_batches(env, net, BATCH_SIZE)):
        reward_mean = float(np.mean(list(map(lambda episode: episode.reward, batch))))
        full_batch, obs, act, reward_bound = filter_batch(full_batch + batch, PERCENTILE)
        if not full_batch:
            continue
        obs_v = torch.FloatTensor(obs)
        act_v = torch.LongTensor(act)
        full_batch = full_batch[-500:]

        optimizer.zero_grad()
        act_score_v = net(obs_v)
        loss_v = objective(act_score_v, act_v)
        loss_v.backward()
        optimizer.step()
        print('%d: loss=%.3f, reward_mean=%.3f, reward_bound=%.3f, batch=%d' %
              (iter_num, loss_v.item(), reward_mean, reward_bound, len(full_batch)))
        writer.add_scalar('loss', loss_v.item(), iter_num)
        writer.add_scalar('reward_mean', reward_mean, iter_num)
        writer.add_scalar('reward_bound', reward_bound, iter_num)
        if reward_mean > 0.8:
            print('Solved!')
            break
    writer.close()




