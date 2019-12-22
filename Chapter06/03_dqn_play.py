import gym
import time
import argparse
import numpy as np
import torch
from Chapter06.lib import wrappers
from Chapter06.lib import dqn_model

DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
FPS = 25

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 预加载模型
    parser.add_argument("-m", "-model", required=True, help="Mpdel file to load")
    # 测试环境
    parser.add_argument("-e", "-env", default=DEFAULT_ENV_NAME, help = "Environment name to use, default=" + DEFAULT_ENV_NAME)
    # 将视频记录保存到空目录下
    parser.add_argument("-r", "-record", help="Directory to store recording")
    args = parser.parse_args()

    env = wrappers.make_env(args.env)
    if args.record:
        env = gym.wrappers.Monitor(env, args.record)
    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n)
    net.load_state_dict(torch.load(args.model))

    state = env.reset()
    total_reward = 0.0
    while True:
        start_ts = time.time()
        env.render()
        state_v = torch.tensor(np.array([state], copy=False))
        q_vals = net(state_v).data.numpy()[0]
        action = np.argmax(q_vals)

        state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
        delta = 1/FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
    print("Total reward: %.2f" % total_reward)


