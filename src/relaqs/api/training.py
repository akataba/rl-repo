from pydoc import locate
from typing import Dict, AnyStr
import numpy as np
import gym
import matplotlib.pyplot as plt
from importlib import import_module
import math
import ray
from ray.tune.registry import register_env

class Training:
    """"
    This class abstracts our the mechanisms of training a model in an environment

    """
    def __init__(self, episodes:int, environment, batch_size:int) -> None:
        self.episodes = episodes
        self.env = environment
        self.state_space = self.env.observation_space.shape[0]
        self.space = self.env.observation_space
        self.action_space = self.env.action_space.n
        self.score_for_episode = []
        self.batch_size = batch_size
        self.reset_target_counter = 0

    # Adaptive learning of Learning Rate
    def learning_rate(t : int , min_rate=0.01 ) -> float  :
        """Decaying learning rate
        Args:
            t: time in a specific episode
        
        """
        return max(min_rate, min(1.0, 1.0 - math.log10((t + 1) / 25)))

    # Decaying exploration rate

    def exploration_rate(t : int, min_rate= 0.1 ) -> float :
        """Decaying exploration rate
        Args:
            t: time in a specific episode
        
        """

        return max(min_rate, min(1, 1.0 - math.log10((t  + 1) / 25)))

    def train_model(self, agent, target_model=False):
        """Trains the model in the enivronment provided

        agent: String that is used to dynamical instatiate the model

        target_model (bool): If true the architecture has a target model that must be reset

        """
        print('Total number of episodes: ', self.episodes)
        for e in range(self.episodes):
            current_state = np.reshape(self.env.reset(), [1, self.state_space])
            done = False
            score = 0 

            while done==False:
                current_action = agent.act(current_state)   # policy action Replace the Q table
                next_state, reward, done, info = self.env.step(current_action)     # increment enviroment
                next_state = np.reshape(next_state, [1, self.state_space])
                agent.remember(current_state, current_action, reward, next_state, done)   # Remember and replay buffer
                current_state = next_state        
                score+=reward        
                self.env.render()     # Render the cartpole environment
                if target_model:
                    self.reset_target_counter +=1
                    if self.reset_target_counter % 5 == 0:
                        agent.reset_target_model()
            self.score_for_episode.append(score)

        if len(agent.memory) > self.batch_size:
            agent.replay_training(self.batch_size)
        self.env.close()

    def plot_performance(self, agent_name:AnyStr):
        """
        Plots the performance of the agent in the environment
        """
        plt.xlabel("Episode")
        plt.ylabel("Score for episode")
        plt.title(agent_name)
        plt.scatter(list(range(len(self.score_for_episode))), self.score_for_episode)
        plt.plot(list(range(len(self.score_for_episode))), self.score_for_episode)
        plt.show()

class TrainRLLib:
    def __init__(self, rllibmodel, env, framework_name="torch", episodes=2) -> None:
        ray.init()
        self.env = env
        register_env("my_env", self.env_creator)
        self.alg_config = rllibmodel()
        self.alg_config.framework(framework_name)
        self.alg_config.environment("my_env", env_config=env.get_default_env_config())
        self.alg = self.alg_config.build()
        self.episodes = episodes

    def env_creator(self, config):
        return self.env(config)

    def train_model(self):
        for _ in range(self.episodes):
            result = self.alg.train()


