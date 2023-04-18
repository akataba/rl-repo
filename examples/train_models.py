# from api.training import Training
# from models.agents import  DDQNAgent
from relaqs.api.training import Training
from relaqs.models.agents import DDQNAgent
import gym
from gym import spaces
import numpy as np
from relaqs.replay_buffers.prioritized_experience import PrioritizedExperienceReplay

# Get Environment
env = gym.make('CartPole-v1')

class CartPoleWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(CartPoleWrapper, self).__init__(env)
        low = self.observation_space.low[2:]
        high = self.observation_space.high[2:]
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        return observation[2:]

env = CartPoleWrapper(env)


buffer = PrioritizedExperienceReplay(0.7,0.6)
agent = DDQNAgent(2,2, replay_buffer=buffer)
train_api = Training(500, env, 32)
train_api.train_model(agent, target_model=True)
train_api.plot_performance('DDQNAgent')