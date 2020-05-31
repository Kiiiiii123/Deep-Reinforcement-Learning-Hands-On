import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = 'FrozenLake-v0'
GAMMA = 0.9
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.rewards = collections.defaultdict(float)
        self.transitions = collections.defaultdict(collections.Counter)
        self.values = collections.defaultdict(float)

    def play_n_random_steps(self, count):
        for _ in range(count):
            action = self.env.action_space.sample()
            new_state, reward, is_done, _ = self.env.step(action)
            self.rewards[(self.state, action, new_state)] = reward
            self.transitions[(self.state, action)][new_state] += 1
            self.state = self.env.reset() if is_done else new_state

    def select_action(self, state):
        best_action, best_value = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or action_value > best_value:
                best_value = action_value
                best_action = action
        return best_action

    def play_episode(self, env):
        total_reward = 0.0
        state = env.reset()
        while True:
            action = self.select_action(state)
            new_state, reward, is_done, _ = env.step(action)
            self.rewards[(state, action, new_state)] = reward
            self.transitions[(state, action)][new_state] += 1
            total_reward += reward
            state = new_state
            if is_done:
                break
        return total_reward

    def value_iteration(self):
        for state in range(self.env.observation_space.n):
            for action in range(self.env.action_space.n):
                action_value = 0.0
                target_counts = self.transitions[(state, action)]
                total = sum(target_counts.values())
                for target_state, count in target_counts.items():
                    reward = self.rewards[(state, action, target_state)]
                    best_action = self.select_action(target_state)
                    val = reward + GAMMA * self.values[target_state, best_action]
                    action_value += (count / total) * val
                self.values[(state, action)] = action_value


if __name__ == '__main__':
    test_env = gym.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment='-v-iteration')

    iter_num = 0
    best_reward = 0.0
    while True:
        iter_num += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES
        writer.add_scalar('reward', reward, iter_num)
        if reward > best_reward:
            print('Best reward updated: %.3f -> %.3f' % (best_reward, reward))
            best_reward = reward
        if reward > 0.80:
            print('Solved in %d iterations!' % iter_num)
            break
    writer.close()






