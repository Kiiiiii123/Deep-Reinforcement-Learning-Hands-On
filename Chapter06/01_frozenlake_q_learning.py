import gym
import collections
from tensorboardX import SummaryWriter

ENV_NAME = 'FrozenLake-v0'
GAMMA = 0.9
ALPHA = 0.2
TEST_EPISODES = 20


class Agent:
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.state = self.env.reset()
        self.value_table = collections.defaultdict(float)



