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


class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor,self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, act_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU()
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.ReLU(),
            nn.Linear(300, 1)
        )

    # 除了输入观察以外还要输入动作
    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))


# 实现用于探索的OU的Agent
class AgentDDPG(ptan.experience.BaseAgent):
    # 传入的参数大多是论文中OU相关的默认值
    def __init__(self, net, device='cpu', ou_enabled=True, ou_mu=0, ou_teta=0.15, ou_sigma=0.2, ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon

    # 当新的episode开始时返回智能体的初始状态
    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = ptan.agent.float32_preprocessor(states).to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()
        # 下面利用OU进程添加探索噪声
        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(shape = action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)






