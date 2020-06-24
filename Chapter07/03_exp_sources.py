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
    s = env.reset()
    print('env.reset() -> %s' % s)
    s = env.step(1)
    print('env.step(1) -> %s' % str(s))
    s = env.step(2)
    print('env.step(2) -> %s' % str(s))

    for _ in range(10):
        r = env.step(0)
        print(r)

    agent = DullAgent(action=1)
    print('agent:', agent([1, 2])[0])

    env = ToyEnv()
    agent = DullAgent(action=1)
    exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=2)
    for idx, exp in enumerate(exp_source):
        if idx > 15:
            break
        print(exp)

    exp_source = ptan.experience.ExperienceSource(env=env, agent=agent, steps_count=4)
    print(next(iter(exp_source)))

    exp_source = ptan.experience.ExperienceSource(env=[ToyEnv(), ToyEnv()], agent=agent, steps_count=2)
    for idx, exp in enumerate(exp_source):
        if idx > 4:
            break
        print(exp)