import gym
import ptan
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


HIDDEN_SIZE = 128


class Net(nn.Module):
    def __init__(self, obs_size, hidden_size, n_actions):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, n_actions)
        )

    def forward(self, x):
        return self.net(x.float())


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    obs_size = env.observation_space.shape[0]
    n_actions = env.action_space.n

    net = Net(obs_size, HIDDEN_SIZE, n_actions)
    tgt_net = ptan.agent.TargetNet(net)
