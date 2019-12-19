import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v0"
GAMMA = 0.9
# 验证value table的效果
TEST_EPISODES = 20


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

    # 相比前面删除了calc_action_value函数

    # 选择某一状态下的最佳动作
    def select_action(self, state):
        best_action, best_value = None, None
        # 查看所有可能动作的Q值
        for action in range(self.env.action_space.n):
            # 直接查询既可以获取状态行为值
            action_value = self.values[(state, action)]
            # 遍历搜索Q值最大的动作
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_action

    # 利用上述动作选择策略进行效果测试，但是也要加入到table中去
    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transits[(state, action)][new_state] += 1
            total_reward += reward
            if is_done:
                break
            state = new_state
        return total_reward

    # 对状态的value table进行迭代
    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transits[(state, action)]
                total = sum(target_counts.values())
                for tgt_state, count in target_counts.items():
                    reward = self.rewards[(state, action, tgt_state)]
                    best_action = self.select_action(tgt_state)
                    action_value += (count/total) * (reward + GAMMA * self.values[(tgt_state, best_action)])
                    self.values[(state, action)] = action_value


if __name__ == "__main__":
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="-v-iteration")

    ite_no = 0
    best_reward = 0.0
    while True:
        ite_no += 1
        # 随机积累数据
        agent.play_n_random_steps(100)
        agent.value_iteration()

        avg_reward = 0.0
        for _ in range(TEST_EPISODES):
            avg_reward += agent.play_episode(test_env)
        # 求测试episode的平均奖励
        avg_reward /= TEST_EPISODES
        writer.add_scalar("reward", avg_reward, ite_no)
        if avg_reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, avg_reward))
            best_reward = avg_reward
        if avg_reward > 0.80:
            print("Solved in %d iterations!" % ite_no)
            break
        writer.close()








