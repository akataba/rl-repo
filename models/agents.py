import random
import numpy as np
import math
from collections import deque
from keras.models import Sequential, Model
from keras.layers import Dense, Input, Lambda, Add
from keras.optimizers import Adam, RMSprop
from api.agent_api import RLAgent
from keras import backend as K


class DDQNAgent(RLAgent):
    def __init__(self, n_state, n_action) -> None:
        super().__init__(n_state, n_action)
        self.online_model = self._build_online_model()
        self.target_model = self._build_target_model(self.online_model.get_weights())
  
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
            self.online_model.fit(current_state, target_f, epochs=1, verbose=0)      

        if self.epsilon > self.epilson_min:
            self.epsilon *= self.epsilon*self.epilson_decay

    def reset_target_model(self):
        weights = self.online_model.get_weights()
        self.target_model.set_weights(weights)


class DQNAgent(RLAgent):
    def __init__(self, n_state, n_action) -> None:
        super().__init(n_state, n_action)
        self.model = self._build_model()        
  
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


class DuelingDDQNAgent(RLAgent):
    def __init__(self, n_state, n_action) -> None:
        super().__init__(n_state, n_action)
        self.online_model = self._build_model()
        self.target_model = self._build_model(weights=self.online_model.get_weights())
    
    def _build_model(self, weights=None):
        state_input = Input(self.n_state)
        layer = state_input 
        layer = Dense(24, input_dim=self.n_state, activation='relu')(layer)
        layer = Dense(24, activation='relu')(layer)

        value_stream = Dense(24, input_dim=self.n_state, activation='relu')(layer)
        advantage_stream = Dense(24, input_dim=self.n_state, activation='relu')(layer)

        value_output  = Dense(1,input_dim=self.n_state, activation='relu')(value_stream)
        value_output = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(self.n_action,))(value_output)

        advantage_output = Dense(self.n_action, input_dim=self.n_state, activation='relu')(advantage_stream)
        advantage_output = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(self.n_action,))(advantage_output)
 
        q_values = Add()([value_output, advantage_output])

        model = Model(inputs = state_input, outputs = q_values)
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        if weights is not None:
            model.set_weight(weights)
        return model
 
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
                greedy_action_with_online = np.amax((self.online_model.predict(next_state)[0])) 
                target =  reward + self.discount*np.amax((self.target_model.predict(next_state)[0][greedy_action_with_online])) 
            target_f = self.online_model.predict(current_state)   
            target_f[0][current_action] = target

            self.online_model.fit(current_state, target_f, epochs=1, verbose=0)      

        if self.epsilon > self.epilson_min:
            self.epsilon *= self.epsilon*self.epilson_decay

    def reset_target_model(self):
        weights = self.online_model.get_weights()
        self.target_model.set_weights(weights)
