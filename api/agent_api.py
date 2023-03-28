from collections import deque
from abc import ABC, abstractmethod

class RLAgent(ABC):
    epsilon = 1
    epilson_decay = 0.995
    epilson_min = 0.01
    learning_rate = 0.001
    memory = deque(maxlen=3000)
    discount = 0.95
    
    def __init__(self, n_state, n_action) -> None:
        self.n_state = n_state
        self.n_action = n_action


    @abstractmethod
    def _build_model():
        NotImplemented

    def remember(self,current_state, current_action, reward, next_state, done):
        self.memory.append((current_state, current_action, reward,next_state, done))

    @abstractmethod
    def act(self, state):
        NotImplemented
    
    @abstractmethod
    def replay_training(self,batch_size):
        NotImplemented
