from pydoc import locate
from typing import Dict, AnyStr
import numpy as np
import gym
import matplotlib.pyplot as plt
from importlib import import_module

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

    def train_model(self, agent, target_model=False):
        """Trains the model in the enivronment provided

        agent: String that is used to dynamical instatiate the model

        target_model (bool): If true the architecture has a target model that must be reset

        """

        for e in range(self.episodes):
            current_state = np.reshape(self.env.reset(), [1, self.state_space])
            done = False
            score = 0 

            while done==False:
                current_action = agent.act(current_state)   # policy action Replace the Q table
                next_state, reward, done, info = self.env.step(current_action)     # increment enviroment
                agent.remember(current_state, current_action, reward, next_state, done)      # Remember and replay buffer
                current_state = next_state        
                score+=reward        
                self.env.render()     # Render the cartpole environment
                if target_model:
                    reset_target_counter +=1
                    if reset_target_counter % 5 == 0:
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

