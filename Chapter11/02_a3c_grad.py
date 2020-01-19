import gym
import ptan
import argparse
from tensorboardX import SummaryWriter

import torch
import torch.nn.utils as nn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

from lib import common

GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_BETA = 0.01

REWARD_STEPS = 4
CLIP_GRAD = 0.1

PROCESSES_COUNT = 4
NUM_ENVS = 15

# BATCH_SIZE用下面两个超参数进行替代
# 定义每个子进程进行loss和gradient计算的batch大小
GRAD_BATCH = 64
# 在单个GPU上进行测试时的选择2，是每次中心进程进行SGD迭代时从子进程获取的gradients batch
TRAIN_BATCH = 2
# 每个优化步骤需要GRAD_BATCH * TRAIN_BATCH个样本

ENV_NAME = 'PongNoFrameskip-v4'
NAME = 'pong'
REWARD_BOUND = 18


def make_env():
    return ptan.common.wrappers.wrap_dqn(gym.make(ENV_NAME))


# 子进程中执行的函数，每个子进程需要执行的操作很多，从搜集数据样本增加到自己计算loss和梯度，有自己的Tensorboard监测数据，train_dequeue将计算的梯度传递给中心进程
def grads_func(proc_name, net, device, train_queue):
    envs = [make_env() for _ in range(NUM_ENVS)]

    agent = ptan.agent.PolicyAgent(lambda x:net(x)[0], device=device, apply_softmax=True)
    exp_source = ptan.experience.ExperienceSourceFirstLast(envs, agent, gamma=GAMMA, steps_count=REWARD_STEPS)

    batch = []
    frame_idx = 0
    writer = SummaryWriter(comment=proc_name)

    with common.RewardTracker(writer, stop_reward=REWARD_BOUND) as tracker:
        with ptan.common.utils.TBMeanTracker(writer, batch_size=100) as tb_tracker:
            for exp in exp_source:
                frame_idx += 1
                new_rewards = exp_source.pop_total_rewards()
                if new_rewards and tracker.reward(new_rewards[0], frame_idx):
                    break

                batch.append(exp)
                if len(batch) < GRAD_BATCH:
                    continue

                states_v, actions_t, vals_ref_v = common.unpack_batch(batch, net, last_val_gamma=GAMMA ** REWARD_STEPS, device=device)
                batch.clear()

                net.zero_grad()
                logits_v, value_v = net(states_v)
                loss_value_v = F.mse_loss(value_v.sequeeze(-1), vals_ref_v)

                log_prob_v = F.log_softmax(logits_v, dim=1)
                adv_v = vals_ref_v - value_v.detach()
                log_prob_actions_v = adv_v * log_prob_v[range(GRAD_BATCH), actions_t]
                loss_policy_v = -log_prob_actions_v.mean()

                prob_v = F.softmax(logits_v, dim=1)
                entropy_loss_v = ENTROPY_BETA * (prob_v * log_prob_v).sum(dim=1).mean()

                loss_v = loss_value_v + entropy_loss_v + loss_policy_v
                # 梯度全部存放在Tensor.grad中用于传递，网络参数同样是共享的。但是梯度在子进程中本地分配
                loss_v.backward()

                tb_tracker.track('advantage', adv_v, frame_idx)
                tb_tracker.track("values", value_v, frame_idx)
                tb_tracker.track("batch_rewards", vals_ref_v, frame_idx)
                tb_tracker.track("loss_entropy", entropy_loss_v, frame_idx)
                tb_tracker.track("loss_policy", loss_policy_v, frame_idx)
                tb_tracker.track("loss_value", loss_value_v, frame_idx)
                tb_tracker.track("loss_total", loss_v, frame_idx)

                # 收集梯度并放到seperate buffer中
                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                grads = [param.grad.data.cpu().numpy() if param.grad is not None else None for param in net.parameters()]
                train_queue.put(grads)

    # 表示训练过程已经结束
    train_queue.put(None)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', default=False, action='store_true', help='Name of the run')
    args = parser.parse_args()
    device = 'cuda' if args.cuda else 'cpu'

    env = make_env()
    # 创建网络并共享权重
    net = common.AtariA2C(env.observation_space.shape, env.action_space.n).to(device)
    net.share_memory()

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, eps=1e-3)
    train_queue = mp.Queue(maxsize=PROCESSES_COUNT)
    data_proc_list = []
    for proc_idx in range(PROCESSES_COUNT):
        proc_name = '-a3c-grad_' + NAME + '_' + args.name + '#%d' % proc_idx
        data_proc = mp.Process(target=grads_func, args=(proc_name, net, device, train_queue))
        data_proc.start()
        data_proc_list.append(data_proc)

    step_idx = 0
    grad_buffer = None

    try:
        while True:
            train_entry = train_queue.get()
            if train_entry is None:
                break

            step_idx += 1
            if grad_buffer is None:
                grad_buffer = train_entry
            else:
                for tgt_grad, grad in zip(grad_buffer, train_entry):
                    tgt_grad += grad

            if step_idx % TRAIN_BATCH == 0:
                for param, grad in zip(net.parameters(), grad_buffer):
                    param.grad = torch.FloatTensor(grad).to(device)

                nn_utils.clip_grad_norm_(net.parameters(), CLIP_GRAD)
                optimizer.step()
                grad_buffer = None

    finally:
        for p in data_proc_list:
            p.terminate()
            p.join()





