import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
# 验证value table的效果
TEST_EPISODE2 = 20

class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        # reward table
        self.rewards = collections.defaultdict(float)
        # state transition table
        # Counter为计算字符个数的计数器
        self.transits = collections.defaultdict(collections.Counter)
        # value table
        self.values = collections.defaultdict(float)

    # 随机探索N步
    def play_n_random_steps(self,count):
        for _ in range(count):
            # 选择随机动作
            action = self.env.action_space.sample()
            # 执行所选择的动作
            new_state, reward, is_done, _ = self.env.step(action)
            # 刷新表格（字典数据结构）
            self.rewards[(self.state, action, new_state)] = reward
            self.transits[(self.state, action)][new_state] += 1
            # 刷新当前状态
            self.state = self.env.reset() if is_done else new_state

    # 计算新的状态行为值
    def cal_action_value(self, state, action):
        target_counts = self.transits[(state, action)]
        # 动作被执行的总次数
        total = sum(target_counts.values())
        action_value = 0
        for tgt_state, count in target_counts.items():
            # 从reward table中提取出奖励值
            reward = self.rewards[(state, action, tgt_state)]
            # 贝尔曼方程
            action_value += (count/total)*(reward + GAMMA * self.values[tgt_state])
        return action_value

    def







