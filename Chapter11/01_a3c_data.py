import gym
import ptan
import numpy as np

import argparse
import collections
from tensorboardX import SummaryWriter

import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
# Pytorch对多进程的封装
import torch.multiprocessing as mp

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01
BATCH_SIZE = 128

REWARD_STEPS = 4
CLIP_GRAD = 0.1

# 预定进程数量
PROCESSES_COUNT = 4
NUM_ENVS = 15

ENV_NAME = "PongNoFrameskip-v4"
NAME = 'pong'
REWARD_BOUND = 18


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))


TotalReward = collections.namedtuple('TotalReward', field_names='reward')


# 在子进程中执行（与抓取、发送样本相关），queue被用来将数据从子进程传送到主进程中，采用多生产者单消费者模式，可以包含两种不同的对象
def data_func(net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]
    agent = ptan.agent.PolicyAgent(lambda x: net(x)[0], device = device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_BOUND)

    for exp in exp_source:
        new_rewards = exp_source.pop_total_rewards()
        if new_rewards:
            train_queue.put(TotalReward(reward=np.mean(new_rewards)))
        train_queue.put(exp)


if __name__ == "__main__":
    # 开启子进程的方法
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action='store_true', help='Enable cuda')
    parser.add_argument("-n", '--name', required=True, help='Name of the run')
    args = parser.parse_args()
    device = 'cuda' if args.cuda else 'cpu'
    writer = SummaryWriter(comment='-a3c-data' + NAME + "_" +args.name)

    env = make_env()
    # 将网络移到cuda设备
    net = common.AtariA2C(env.observation_space.shape, env.action_space.n).to(device)
    # 共享网络权重
    net.share_memory()
    optimizer = optim.Adam(net.parameters(), lr=, eps=1e-3)

    # 创建用于向我们传递数据的队列
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for _ in range(PROCESSES_COUNT):
        # 开始不断开启子进程
        data_proc = mp.Process(target=data_func, args=(net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    batch = []
    step_idx = 0

    try:
        with common.RewardTracker(writer, stop_reward=REWARD_BOUND) as tracker:
            with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
                while True:
                    # 从队列获取并处理TotalReward对象和experience
                    train_entry = train_queue.get()
                    if isinstance(train_entry, TotalReward):
                        if tracker.reward(train_entry.reward, step_idx):
                            break
                        continue

                    step_idx += 1
                    batch.append(train_entry)
                    if len(batch) < BATCH_SIZE:
                        continue

                    states_v, actions_t, vals_ref_v = common.unpack_batch(batch, net, last_val_gamma = GAMMA ** REWARD_STEPS, device=device)
                    batch.clear()

                    optimizer.zero_grad()
                    logits_v, value_v= net(states_v)
                    loss_value_v = F.mse_loss(value_v.squeeze(-1), vals_ref_v)

                    log_prob_v = F.log_softmax(logits_v, dim=1)
                    adv_v = vals_ref_v - value_v.detach()
                    log_prob_actions_v = adv_v * log_prob_v[range(BATCH_SIZE), actions_t]
                    loss_policy_v = -log_prob_actions_v.mean()

                    prob_v = F.softmax(logits_v, dim=1)
                    entropy_loss_v = ENTROPY_BETA*(prob_v * log_prob_v).sum(dim=1).mean()

                    loss_v = entropy_loss_v + loss_policy_v + loss_value_v
                    loss_v.backward()
                    nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                    optimizer.step()

                    tb_tracker.track("advantage", adv_v, step_idx)
                    tb_tracker.track("values", value_v, step_idx)
                    tb_tracker.track("batch_rewards", vals_ref_v, step_idx)
                    tb_tracker.track("loss_entropy", entropy_loss_v, step_idx)
                    tb_tracker.track("loss_policy", loss_policy_v, step_idx)
                    tb_tracker.track("loss_value", loss_value_v, step_idx)
                    tb_tracker.track("loss_total", loss_v, step_idx)
    finally:
        # 程序中断或问题解决时执行
        for p in data_proc_list:
            # 结束运行子进程并等待
            p.terminate()
            p.join()






