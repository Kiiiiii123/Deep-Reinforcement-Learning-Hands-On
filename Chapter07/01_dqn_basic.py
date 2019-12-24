import gym
import ptan
import argparse

import torch
import torch.optim as optim

from tensorboardX import SummaryWriter

from lib import dqn_model, common

if __name__ == "__main__":
    params = common.HYPERPARAMS['pong']

    # 根据命令行选择GPU
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    # 创建游戏环境并进行封装
    env = gym.make(params['env_name'])
    env = ptan.common.wrappers.wrap_dqn(env)

    # 创建两大主网络
    writer = SummaryWriter(comment="-" + params['run_name'] + "-basic")
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    # 十分简单的创建目标网络，包括复制权重和分时段同步
    tgt_net = ptan.agent.TargetNet(net)

    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    # 创建智能体，通过网络选择动作
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    # 记录采样经验
    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])

    # 优化器以及训练迭代计数
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])
    frame_idx = 0

    # 每个episode结束后报告平均奖励值
    with common.RewardTracker(writer, params['stop_reward']) as reward_tracker:
        while True:
            frame_idx += 1
            # 将前面的实现在这里封装简化了
            buffer.populate(1)
            epsilon_tracker.frame(frame_idx)

            # 获取已经完成的episode的奖励的列表并发送给reward tracker以检测是否结束训练
            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            # 填充replay buffer之后才开始进行训练
            if len(buffer) < params['replay_initial']:
                continue

            # 随机梯度下降最优化
            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], device=device)
            loss_v.backward()
            optimizer.step()

            # 延迟更新目标网络参数
            if frame_idx % params['target_net_sync'] == 0:
                tgt_net.sync()












