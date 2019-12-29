import gym
import ptan
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common

# n-step
REWARD_STEPS = 2

# priority replay
PRIO_REPLAY_ALPHA = 0.6
BETA_START = 0.4
BETA_FRAMES = 100000

# C51
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)


class RainbowDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(RainbowDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self. _get_conv_out(input_shape)
        # 预测输入状态的值分布
        self.fc_val = nn.Sequential(
            dqn_model.NoisyLinear(conv_out_size, 256),
            nn.ReLU(),
            dqn_model.NoisyLinear(256, N_ATOMS)
        )

        # 预测游戏中每个动作的分布
        self.fc_adv = nn.Sequential(
            dqn_model.NoisyLinear(conv_out_size, 256),
            nn.ReLU(),
            dqn_model.NoisyLinear(256, n_actions * N_ATOMS)
        )

        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
         o = self.conv(torch.zeros(1, *shape))
         return int(np.prod(o.size()))

    # 获取动作值分布
    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        fx = fx.type(torch.FloatTensor).cuda()
        conv_out = self.conv(fx).view(batch_size, -1)
        val_out = self.fc_val(conv_out).view(batch_size, -1, N_ATOMS)
        adv_out = self.fc_adv(conv_out).view(batch_size, -1, N_ATOMS)
        adv_mean = adv_out.mean(dim=1, keepdim=True)
        return val_out + adv_out - adv_mean

    # 对输出的概率分布进行softmax
    def apply_softmax(self, t):
        return self.softmax(t.view(-1, N_ATOMS).view(t.size()))

    def both(self, x):
        x = x.type(torch.FloatTensor).cuda()
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        res = weights.sum(dim=2)
        return cat_out, res

    def qvals(self, x):
        return self.both(x)[1]


def calc_loss(batch, batch_weights, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = common.un_pack(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    # 加入权重以进行优先级采样
    batch_weights_v = torch.tensor(batch_weights).to(device)

    # Double DQN方法，主网络选择动作，目标网络获取值
    # 以前调用两次以计算网络输出，在GPU上并不高效，这里将当前状态与下一状态一起传入
    distr_v, qvals_v = net.both(torch.cat((states_v, next_states_v)))
    next_qvals_v = qvals_v[batch_size:]
    distr_v = distr_v[:batch_size]

    # 根据主网络选择下一个状态下的动作
    next_actions_v = next_qvals_v.max(1)[1]
    # 根据目标网络获得状态行为值分布
    next_distr_v = tgt_net(next_states_v)
    next_best_distr_v = next_distr_v[range(batch_size), next_actions_v.data]
    next_best_distr_v = tgt_net.apply_spftmax(next_best_distr_v)
    next_best_distr = next_best_distr_v.data.cpu().numpy()

    # 转到CPU之后进行投影
    dones = dones.astype(np.bool)
    proj_distr = common.distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)

    # 计算KL散度
    proj_distr_v = torch.tensor(proj_distr)
    loss_v = -state_log_sm_v * proj_distr_v
    loss_v = batch_weights * loss_v.sum(dim=1)
    return loss_v.mean(), loss_v + 1e-5


if __name__ == "__main__":
    params = common.HYPERPARAMS['pong']
    params['epsilon_frames'] *= 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + params["run_name"] + "-rainbow")
    net = RainbowDQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = ptan.agent.TargetNet(net)
    agent = ptan.agent.DQNAgent(lambda x:net.qvals(x), ptan.actions.ArgmaxActionSelector(), device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params["gamma"], steps_count=REWARD_STEPS)
    buffer = ptan.experience.PrioritizedReplayBuffer(exp_source, params["replay_size"], PRIO_REPLAY_ALPHA)
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    beta = BETA_START

    with common.RewardTracker(writer, params["stop_reward"]) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            beta = min(1.0, BETA_START + frame_idx * (1 - BETA_START) / BETA_FRAMES)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            optimizer.zero_grad()
            # 每个样本按优先级采样并附带权重
            batch, batch_indices, batch_weights = buffer.sample(params['batch_size'], beta)
            # n-steps
            loss_v, sample_prios_v = calc_loss(batch, batch_weights, net, tgt_net.model, params['gamma'] ** REWARD_STEPS, device=device)
            loss_v.backward()
            optimizer.step()
            # 根据损失值更新采样优先级
            buffer.update_priorities(batch_indices, sample_prios_v.data.cpu().numpy())

            # 更新目标网络
            if frame_idx % params["target_net_sync"] == 0:
                tgt_net.sync()


