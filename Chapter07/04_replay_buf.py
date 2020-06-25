import gym
import ptan
from typing import List, Tuple, Any, Optional


class ToyEnv(gym.Env):
    def __init__(self):
        super(ToyEnv, self).__init__()
        self.observation_space = gym.spaces.Discrete(n=5)
        self.action_space = gym.spaces.Discrete(n=3)
        self.step_index = 0

    def reset(self):
        self.step_index = 0
        return self.step_index

    def step(self, action):
        is_done = self.step_index == 10
        if is_done:
            return self.step_index % self.observation_space.n, 0.0, is_done, {}
        self.step_index += 1
        return self.step_index % self.observation_space.n, float(action), self.step_index == 10, {}


class DullAgent(ptan.agent.BaseAgent):
    """
    Agent always returns the fixed action
    """
    def __init__(self, action: int):
        self.action = action

    def __call__(self, observations: List[Any], state: Optional[List] = None) -> Tuple[List[int], Optional[List]]:
        return [self.action for _ in observations], state


if __name__ == '__main__':
    env = ToyEnv()
    agent = DullAgent(action=1)
    exp_source = ptan.experience.ExperienceSourceFirstLast(agent=agent, env=env, gamma=1.0, steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=100)

    for step in range(6):
        buffer.populate(1)
        if len(buffer) < 5:
            continue
        batch = buffer.sample(4)
        print('Train time: %d batch samples:' % len(batch))
        for s in batch:
            print(s)