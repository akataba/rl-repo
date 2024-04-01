import random
import numpy as np
from relaqs.environments import NoisySingleQubitEnv
from relaqs.api.gates import RandomSU2

class ChangingTargetEnv(NoisySingleQubitEnv):
    @classmethod
    def get_default_env_config(cls):
        config_dict = super().get_default_env_config()
        config_dict["observation_space_size"] = 67
        config_dict["U_target_list"] = []
        config_dict["target_generation_function"] = RandomSU2
        return config_dict
    
    def __init__(self, env_config):
        super().__init__(env_config)
        self.U_target_list = env_config["U_target_list"]
        self.target_generation_function = env_config["target_generation_function"]

    def set_target_gate(self):
        if len(self.U_target_list) == 0:
            U = self.target_generation_function().get_matrix()
        else:
            U = random.choice(self.U_target_list)
        self.U_target = self.unitary_to_superoperator(U)

    def reset(self, *, seed=None, options=None):
        _, info = super().reset()
        self.set_target_gate()
        starting_observation = self.get_observation()
        return starting_observation, info

    def get_observation(self):
        observation = super().get_observation()
        return np.append(observation, self.unitary_to_observation(self.U_target))
