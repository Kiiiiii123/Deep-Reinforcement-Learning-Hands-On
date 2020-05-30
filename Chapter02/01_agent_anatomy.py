# Python type annotation
from typing import List
import random


class Environment:
    def __init__(self):
        # limit the number of time steps that the agent is allowed to take to interact with the environment
        self.steps_left = 10

    # return the current environment's observation to the agent
    def get_observation(self) -> List[float]:
        return [0.0, 0.0, 0.0]

    # allow the agent to query the set of actions it can execute
    def get_actions(self) -> List[int]:
        return [0, 1]

    # signal the end of the episode to the agent
    def is_done(self) -> bool:
        return self.steps_left == 0

    # handle an agent's action and return the reward for this action.
    def action(self, action: int) -> float:
        if self.is_done():
            raise Exception('Game is over!')
        self.steps_left -= 1
        return random.random()


class Agent:
    def __init__(self):
        # initialize the counter that will keep the total reward accumulated by the agent during the episode
        self.total_reward = 0.0

    # accept the environment instance as an argument and allow the agent to perform the actions
    def step(self, env: Environment):
        current_obs = env.get_observation()
        actions = env.get_actions()
        reward = env.action(random.choice(actions))
        self.total_reward += reward


if __name__ == '__main__':
    env = Environment()
    agent = Agent()

    while not env.is_done():
        agent.step(env)

    print('Total reward got: %.4f' % agent.total_reward)


