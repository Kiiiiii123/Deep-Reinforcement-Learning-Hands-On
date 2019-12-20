import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
# Q-value更新的学习率
ALPHA = 0.2
TEST_SPISODES = 20

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.values = collections.defaultdict(float)

    # 与环境进行交互并采样元组
    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_state, reward, is_done, _ = self.env.step(action)
        self.state = self.env.reset() if is_done else new_state
        return (old_state, action, reward, new_state)

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values.get([state, action])
            if best_value is None or action_value > best_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    # Q-table的更新
    def value_update(self, s, a, r, nest_s):
        best_v, _ = self.best_value_and_action(nest_s)
        new_val = r + GAMMA * best_v
        old_val = self.values[(s, a)]
        self.values[(s, a)] = old_val * (1 - ALPHA) + new_val * ALPHA

    # 实战
    def play_episode(self, env):
         total_reward = 0.0
         state = self.env.reset()
         while True:
             _, action = self.best_value_and_action(state)
             new_state, reward, is_done, _ = self.env.step(action)
             total_reward += reward
             if is_done:
                 break
             state = new_state
         return total_reward

if __name__ == "__main__":

