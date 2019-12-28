import gym
import ptan
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import common

SAVE_STATES_IMG = False
SAVE_TRANSITIONS_IMG = False

# 开启以保存概率分布简化Debug，训练过程可视化
if SAVE_STATES_IMG or SAVE_TRANSITIONS_IMG:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pylab as plt

# 设置值分布的范围，柱状体数量，计算之间的间距
Vmax = 10
Vmin = -10
N_ATOMS = 51
DELTA_Z = (Vmax - Vmin) / (N_ATOMS - 1)

# 拿出多少状态进行平均值计算以及多久更新一次
STATES_TO_EVALUATE = 1000
EVAL_EVERY_FRAME = 100


class DistributionalDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DistributionalDQN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8 ,stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            # 全连接层输出部分变为包含n_actions*N_ATOMS个元素的矩阵，包含每个动作额概率分布
            nn.Linear(512, n_actions * N_ATOMS)
        )

        # 注册不应被视为模型参数的缓冲区，_buffer中的元素不会被优化器更新，如果在模型中需要需要一些参数，并且要通过state_dict返回，且不需要被优化器训练，那么这些参数可以注册在_buffer中
        # torch.arange函数不包含末尾的按照指定间隔生成列表
        self.register_buffer("supports", torch.arange(Vmin, Vmax+DELTA_Z, DELTA_Z))
        # 对n维输入张量运用Softmax函数，将张量的每个元素缩放到（0,1）区间且和为1，dim=1表示按行计算
        self.softmax = nn.Softmax(dim=1)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        batch_size = x.size()[0]
        fx = x.float() / 256
        fx = fx.type(torch.FloatTensor).cuda()
        conv_out = self.conv(fx).view(batch_size, -1)
        fc_out = self.fc(conv_out)
        return fc_out.view(batch_size, -1, N_ATOMS)

    # 对输出应用softmax函数并保持张量形式
    def apply_softmax(self, t):
        # 只针对单个分布应用softmax，因为求的是(state,action)对的奖励值，view函数起到的是reshape的作用
        return self.softmax(t.view(-1, N_ATOMS)).view(t.size())

    # 从概率分布中获取Q-values，奖励期望值
    def both(self, x):
        cat_out = self(x)
        probs = self.apply_softmax(cat_out)
        weights = probs * self.supports
        # 按列相加求和
        res = weights.sum(dim=2)
        return cat_out, res

    # 提取Q-values
    def qvals(self, x):
        return self.both(x)[1]


def calc_values_of_states(states, net, device="cpu"):
    mean_vals = []
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_values_v = net.qvals(states_v)
        best_action_values_v = action_values_v.max(1)[0]
        mean_vals.append(best_action_values_v.mean().item())
    return np.mean(mean_vals)


# 保存状态概率分布图
def save_state_images(frame_idx, states, net, device="cpu", max_states=200):
    ofs = 0
    p = np.arange(Vmin, Vmax + DELTA_Z, DELTA_Z)
    for batch in np.array_split(states, 64):
        states_v = torch.tensor(batch).to(device)
        action_prob = net.apply_softmax(net(states_v)).data.cpu().numpy()
        batch_size, num_actions, _ = action_prob.shape
        for batch_idx in range(batch_size):
            plt.clf()
            for action_idx in range(num_actions):
                plt.subplot(num_actions, 1, action_idx+1)
                plt.bar(p, action_prob[batch_idx, action_idx], width=0.5)
            plt.savefig("states/%05d_%08d.png" % (ofs + batch_idx, frame_idx))
        ofs += batch_size
        if ofs >= max_states:
            break


# 保存概率分布投影效果
def save_transition_images(batch_size, predicted, projected, next_distr, dones, rewards, save_prefix):
    for batch_idx in range(batch_size):
        is_done = dones[batch_idx]
        reward = rewards[batch_idx]
        plt.clf()
        p = np.arange(Vmin, Vmax + DELTA_Z, DELTA_Z)
        plt.subplot(3, 1, 1)
        plt.bar(p, predicted[batch_idx], width=0.5)
        plt.title("Predicted")
        plt.subplot(3, 1, 2)
        plt.bar(p, projected[batch_idx], width=0.5)
        plt.title("Projected")
        plt.subplot(3, 1, 3)
        plt.bar(p, next_distr[batch_idx], width=0.5)
        plt.title("Next state")
        suffix = ""
        if reward != 0.0:
            suffix = suffix + "_%.0f" % reward
        if is_done:
            suffix = suffix + "_done"
        plt.savefig("%s_%02d%s.png" % (save_prefix, batch_idx, suffix))


def calc_loss(batch, net, tgt_net, gamma, device="cpu", save_prefix=None):
    states, actions, rewards, dones, next_states = common.unpack_batch(batch)
    batch_size = len(batch)

    states_v = torch.tensor(states).to(device)
    actions_v = torch.tensor(actions).to(device)
    next_states_v = torch.tensor(next_states).to(device)

    # next state distribution
    next_distr_v, next_qvals_v = tgt_net.both(next_states_v)
    # 贪心策略选择动作用于与DQN形成对照组
    next_actions = next_qvals_v.max(1)[1].data.cpu().numpy()
    next_distr = tgt_net.apply_softmax(next_distr_v).data.cpu().numpy()

    # 最佳动作下的概率分布
    next_best_distr =next_distr[range(batch_size), next_actions]
    dones = dones.astype(np.bool)

    # 对概率分布进行投影，投影出的概率分布是期望网络输出的目标概率分布
    proj_distr = common.distr_projection(next_best_distr, rewards, dones, Vmin, Vmax, N_ATOMS, gamma)

    # 求网路输出并计算两个分布之间的KL散度
    distr_v = net(states_v)
    # log_softmax函数在softmax函数的基础上在进行log操作
    state_action_values = distr_v[range(batch_size), actions_v.data]
    state_log_sm_v = F.log_softmax(state_action_values, dim=1)
    proj_distr_v = torch.tensor(proj_distr).to(device)

    # 每一帧保存状态的概率分布投影过程
    if save_prefix is not None:
        pred = F.softmax(state_action_values, dim=1).data.cpu().numpy()
        save_transition_images(batch_size, pred, proj_distr, next_best_distr, dones, rewards, save_prefix)

    loss_v = -state_log_sm_v * proj_distr_v
    return loss_v.sum(dim=1).mean()


if __name__ == "__main__":
    params = common.HYPERPARAMS['pong']
#    params['epsilon_frames'] *= 2
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    writer = SummaryWriter(comment="-" + params['run_name'] + "-distrib")
    net = DistributionalDQN(env.observation_space.shape, env.action_space.n).to(device)

    tgt_net = ptan.agent.TargetNet(net)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    # 智能体计算Q-values的方式变了，因此需要改动
    agent = ptan.agent.DQNAgent(lambda x: net.qvals(x), selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    # 取出用于评估的状态集
    eval_states = None
    prev_save = 0
    save_prefix = None

    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                continue

            if eval_states is None:
                eval_states = buffer.sample(STATES_TO_EVALUATE)
                eval_states = [np.array(transition.state, copy=False) for transition in eval_states]
                eval_states = np.array(eval_states, copy=False)

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])

            # 文件保存前缀
            save_prefix = None
            # 是否保存概率分布投影过程
            if SAVE_TRANSITIONS_IMG:
                interesting = any(map(lambda s: s.last_state is None or s.reward != 0.0, batch))
                if interesting and frame_idx // 30000 > prev_save:
                    save_prefix = "images/img_%08d" % frame_idx
                    prev_save = frame_idx // 30000

            loss_v = calc_loss(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device, save_prefix=save_prefix)
            loss_v.backward()
            optimizer.step()

            # 目标网络同步
            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()

            # 状态评估
            if frame_idx % EVAL_EVERY_FRAME == 0:
                mean_val = calc_values_of_states(eval_states, net, device=device)
                writer.add_scalar("values_mean", mean_val, frame_idx)

            # 每一万帧保存一次状态概率分布图
            if SAVE_STATES_IMG and frame_idx % 10000 == 0:
                save_state_images(frame_idx, eval_states, net, device=device)




