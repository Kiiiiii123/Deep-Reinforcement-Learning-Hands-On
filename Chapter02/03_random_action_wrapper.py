import gym
from typing import TypeVar
import random

Action = TypeVar('Action')


class RandomActionWrapper(gym.ActionWrapper):
    # initialize our wrapper by calling a parent's __init__ method and saving epsilon
    def __init__(self, env, epsilon=0.1):
        super(RandomActionWrapper, self).__init__(env)
        self.epsilon = epsilon

    # override from a parent's class to tweak the agent's actions
    def action(self, action: Action) -> Action:
        if random.random() < self.epsilon:
            print('Random!')
            return self.env.action_space.sample()
        return action


if __name__ == '__main__':
    # create a normal CartPole environment and pass it to our Wrapper constructor
    env = RandomActionWrapper(gym.make('CartPole-v0'))
    total_reward = 0.0
    obs = env.reset()

    while True:
        obs, reward, dome, _ = env.step(0)
        total_reward += reward
        if dome:
            break

    print('Reward got: %.2f' % total_reward)