import numpy as np
from relaqs.environments import NoisySingleQubitEnv
from relaqs.api.gates import RandomSU2

class ChangingTargetEnv(NoisySingleQubitEnv):
    @classmethod
    def get_default_env_config(cls):
        config_dict = super().get_default_env_config()
        config_dict["observation_space_size"] = 67
        return config_dict

    def reset(self, *, seed=None, options=None):
        _, info = super().reset()
        U = RandomSU2().get_matrix()
        self.U_target = U
        self.superoperator_target = self.unitary_to_superoperator(U)
        starting_observation = self.get_observation()
        print(len(starting_observation))
        return starting_observation, info
    
    def get_observation(self):
        observation = super().get_observation()
        return np.append(observation, self.unitary_to_observation(self.U_target))