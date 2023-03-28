import random
import gym
import numpy as np
import math
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import tensorflow as tf

# Set parameters
env = gym.make('CartPole-v1')

state_space = env.observation_space.shape[0]
space = env.observation_space
action_space = env.action_space.n
batch_size = 5
n_episodes = 3
score_for_episode = []


class DQNAgent:
    def __init__(self, n_state, n_action) -> None:
        self.n_state = n_state
        self.n_action = n_action
        self.epsilon = 1
        self.epilson_decay = 0.995
        self.epilson_min = 0.01
        self.learning_rate = 0.001
        self.memory = deque(maxlen=3000)
        self.model = self._build_model()
        self.discount = 0.95
        
  
    def _build_model(self):
        # build the keras model
        model = Sequential()
        model.add(Dense(24, input_dim=self.n_state, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.n_action, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self,current_state, current_action, reward, next_state, done):
        self.memory.append((current_state, current_action, reward,next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_action)
        actions = self.model.predict(state)
        return np.argmax(actions[0])

    def replay_training(self,batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for current_state, current_action, reward, next_state, done in minibatch:
            new_q_value = reward
            if not done:
                new_q_value =  reward + self.discount*np.amax((self.model.predict(next_state)[0]))
            new_q_value_f = self.model.predict(current_state)
            new_q_value_f[0][current_action] = new_q_value
            self.model.fit(current_state, new_q_value_f, epochs=1, verbose=0)

        if self.epsilon > self.epilson_min:
            self.epsilon *= self.epsilon*self.epilson_decay

dq_network = DQNAgent(state_space, action_space)

for e in range(n_episodes):
    
    # Siscretize state into buckets
    current_state = np.reshape(env.reset(), [1, state_space])
    done = False
    score = 0 
    step = 0 
    
    while done==False: 
        current_action = dq_network.act(current_state)   # policy action Replace the Q table
        next_state, reward, done, info = env.step(current_action)     # increment enviroment
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_space])
        dq_network.remember(current_state, current_action, reward, next_state, done)      # Remember and replay buffer
        current_state = next_state        
        score+=reward        
        env.render()     # Render the cartpole environment
    score_for_episode.append(score)
    if len(dq_network.memory) > batch_size:
        dq_network.replay_training(batch_size)
     
env.close()

# plt.xlabel("Episode")
# plt.ylabel("Score for episode")
# plt.title("Scoring DQN Agent")
# plt.scatter(list(range(len(score_for_episode))), score_for_episode)
# plt.plot(list(range(len(score_for_episode))), score_for_episode)
# plt.show()