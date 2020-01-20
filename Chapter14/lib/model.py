import ptan
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

HID_SIZE = 128


class ModelA2C(nn.Module):
    def __init__(self, ob_size, act_size):
        super(ModelA2C, self).__init__()

        self.base = nn.Sequential(
            nn.Linear(ob_size, HID_SIZE),
            nn.ReLU(),
        )
        self.mu = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Tanh(),
        )
        self.var = nn.Sequential(
            nn.Linear(HID_SIZE, act_size),
            nn.Softplus(),
        )
        self.value = nn.Linear(HID_SIZE, 1)

    def forward(self, x):
        base_out = self.base(x)
        return self.mu(base_out), self.var(base_out), self.value(base_out)


# 将观察转换到动作
class AgentA2C(ptan.agent.BaseAgent):
    def __init__(self, net, device='cpu'):
        self.net = net
        self.device = device

        def __call__(self, states, agent_states):
            states_v = ptan.agent.float32_preprocessor(states).to(device)
            mu_v, var_v, _= net(states_v)
            mu = mu_v.cpu().numpy()
            sigma = torch.sqrt(var_v).data.cpu().numpy()
            actions = np.random.normal(mu, sigma)
            actions = np.clip(actions, -1, 1)
            return actions, agent_states



