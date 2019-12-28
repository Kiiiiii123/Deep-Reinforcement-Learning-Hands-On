import sys
import time
import numpy as np
import torch
import torch.nn as nn

# Pong环境的超参数设置
HYPERPARAMS = {
    'pong': {
        'env_name': "PongNoFrameskip-v4",
        'stop_reward': 18.0,
        'run_name': 'pong',
        'replay_size': 100000,
        'replay_initial': 10000,
        'target_net_sync': 1000,
        'epsilon_frames': 10**5,
        'epsilon_start': 1.0,
        'epsilon_final': 0.02,
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'batch_size': 32
    },
}


# 将样本数据解包到numpy数组
def unpack_batch(batch):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        state = np.array(exp.state, copy=False)
        states.append(state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(state)
        else:
            last_states.append(np.array(exp.last_state, copy=False))
    return np.array(states, copy=False), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.uint8), np.array(last_states, copy=False)


# 计算损失函数
def calc_loss_dqn(batch, net, tgt_net, gamma, device="cpu"):
    states, actions, rewards, dones, next_states = unpack_batch(batch)

    # 转换为张量
    states_v = torch.tensor(states).to(device)
    next_states_v = torch.tensor(next_states).to(device)
    actiosn_v = torch.tensor(actions).to(device)
    rewards_v = torch.tensor(rewards).to(device)
    done_mask = torch.ByteTensor(dones).to(device)

    # 计算预测值和目标值
    state_action_values = net(states_v).gather(1, actiosn_v.unsqueeze(-1)).squeeze(-1)
    next_state_values = tgt_net(next_states_v).max(1)[0]
    next_state_values[done_mask] = 0.0

    # 计算损失值
    expected_state_action_values = next_state_values.detach() * gamma + rewards_v
    return nn.MSELoss()(state_action_values, expected_state_action_values)


# 介绍两个简化训练循环的工具
# 执行epsilon的线性衰减策略
class EpsilonTracker:
    def __init__(self, epsilon_greedy_selector, params):
        self.epsilon_greedy_selector = epsilon_greedy_selector
        self.epsilon_start = params['epsilon_start']
        self.epsilon_final = params['epsilon_final']
        self. epsilon_frames = params['epsilon_frames']
        self.frame(0)

    def frame(self, frame):
        self.epsilon_greedy_selector.epsilon = max(self.epsilon_final, self.epsilon_start - frame / self.epsilon_frames)


# 获取每个episode结束时的奖励，以及最近episodes的平均奖励
class RewardTracker:
    def __init__(self, writer, stop_reward):
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    # 当前迭代次数，帧处理速度，epsilon值，平均奖励值，
    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        speed = (frame - self.ts_frame) / (time.time() - self.ts)
        self.ts_frame = frame
        self.ts = time.time()
        mean_reward = np.mean(self.total_rewards[-100:])
        epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
        print("%d: done %d games, mean reward %.3f, speed %.2f f/s%s" % (frame, len(self.total_rewards), mean_reward, speed, epsilon_str))
        sys.stdout.flush()
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
            self.writer.add_scalar("speed", speed, frame)
            self.writer.add_scalar("reward_100", mean_reward, frame)
            self.writer.add_scalar("reward", reward, frame)
            if mean_reward > self.stop_reward:
                print("Solved in %d frames!" % frame)
                return True
            return False


# 将下一个状态的分布进行投影
def distr_projection(next_distr, rewards, dones, Vmin, Vmax, n_atoms, gamma):
    batch_size = len(rewards)
    # 用于存放概率分布的矩阵
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    # 柱状体之间的间距
    delta_z = (Vmax - Vmin) / (n_atoms - 1)

    for atom in range(n_atoms):
        # 第一步进行边界裁切
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards + (Vmin + atom * delta_z) * gamma))

        # 结果带小数点的运算结果
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]

        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones.any():
        # 已经来到epside末尾的样例将分布置为0
        proj_distr[dones] = 0.0
        tz_j = np.minimum(Vmax, np.maximum(Vmin, rewards[dones]))
        b_j = (tz_j - Vmin) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        # 处于episode完结处的样本再次进行划分
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        # 当可迭代对象中有任意一个不为False，则返回True
        if eq_dones.any():
            proj_distr[eq_dones, l] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u] = (b_j - l)[ne_mask]
    return proj_distr


