import random
import gym
from gym import spaces
import numpy as np
import math
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt



# Set parameters
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
state_space = env.observation_space.shape[0]
space = env.observation_space
action_space = env.action_space.n
episode_time = 501
batch_size = 32
n_episodes = 500
reset_target_time = 5
reset_target_counter = 0
score_for_episode = []

class DDQNAgent:
    def __init__(self, n_state, n_action) -> None:
        self.n_state = n_state
        self.n_action = n_action
        self.epsilon = 1
        self.epilson_decay = 0.995
        self.epilson_min = 0.01
        self.learning_rate = 0.001
        self.memory = deque(maxlen=3000)
        self.online_model = self._build_online_model()
        self.target_model = self._build_target_model(self.online_model.get_weights())
        self.discount = 0.95

  
    def _build_online_model(self):
        # build the keras model
        model = Sequential()
        model.add(Dense(24, input_dim=self.n_state, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.n_action, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def _build_target_model(self, weights):
        # build the keras model
        model = Sequential()
        model.add(Dense(24, input_dim=self.n_state, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.n_action, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        model.set_weights(weights)
        return model


    def remember(self,current_state, current_action, reward, next_state, done):
        self.memory.append((current_state, current_action, reward,next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.n_action)
        actions = self.online_model.predict(state)
        return np.argmax(actions[0])

    def replay_training(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for current_state, current_action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target =  reward + self.discount*np.amax((self.target_model.predict(next_state)[0])) 
            target_f = self.online_model.predict(current_state)   
            target_f[0][current_action] = target
            # online_q_value = self.online_model.predict(current_state)
            self.online_model.fit(current_state, target_f, epochs=1, verbose=0)      

        if self.epsilon > self.epilson_min:
            self.epsilon *= self.epsilon*self.epilson_decay

    def reset_target_model(self):
        weights = self.online_model.get_weights()
        self.target_model.set_weights(weights)


ddq_network = DDQNAgent(state_space, action_space)

for e in range(n_episodes):
    
    # Siscretize state into buckets
    current_state = np.reshape(env.reset(), [1, state_space])
    done = False
    score = 0 
    step = 0 
    for t in range(episode_time):
        current_action = ddq_network.act(current_state)   # policy action Replace the Q table
        next_state, reward, done, info = env.step(current_action)     # increment enviroment
        # reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_space])
        ddq_network.remember(current_state, current_action, reward, next_state, done)      # Remember and replay buffer
        current_state = next_state        
        score+=reward        
        env.render()     # Render the cartpole environment
        reset_target_counter +=1
        if reset_target_counter % 5 == 0:
            ddq_network.reset_target_model()
        if done:
            break

    score_for_episode.append(score)
    if len(ddq_network.memory) > batch_size:
        ddq_network.replay_training(batch_size)
     
env.close()

plt.xlabel("Episode")
plt.ylabel("Score for episode")
plt.title("Scoring DDQN Agent")
plt.scatter(list(range(len(score_for_episode))), score_for_episode)
plt.plot(list(range(len(score_for_episode))), score_for_episode)
plt.show()      