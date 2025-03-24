import random
import numpy as np
from relaqs.environments import SingleQubitEnv, NoisySingleQubitEnv
from relaqs.api.gates import RandomSU2
from relaqs.api.utils import normalize

class ChangingTargetEnv(SingleQubitEnv):
    @classmethod
    def get_default_env_config(cls):
        config_dict = super().get_default_env_config()
        config_dict["U_target_list"] = []
        config_dict["target_generation_function"] = RandomSU2
        return config_dict
    
    def __init__(self, env_config):
        super().__init__(env_config)
        self.U_target_list = env_config["U_target_list"]
        self.target_generation_function = env_config["target_generation_function"]

    def set_target_gate(self):
        if len(self.U_target_list) == 0:
            self.U_target= self.target_generation_function().get_matrix()
        else:
            self.U_target = random.choice(self.U_target_list.get_matrix())
        self.U_target_dm = self.U_target.copy()

    def set_initial_gate(self):
        self.U_initial = self.target_generation_function().get_matrix()
        self.U_initial_dm = self.U_initial.copy()

    def reset(self, *, seed=None, options=None):
        _, info = super().reset()
        self.set_target_gate()
        self.set_initial_gate()
        starting_observation = self.get_observation()
        return starting_observation, info

    def return_env_config(self):
        env_config = super().get_default_env_config()
        env_config.update({
            "observation_space_size": 8,
            "num_Haar_basis": self.num_Haar_basis,
            "steps_per_Haar": self.steps_per_Haar,
            "verbose": self.verbose,
            "U_init": self.U_initial,
            "U_target": self.U_target,
            "target_generation_function": self.target_generation_function,
            "U_target_list": self.U_target_list,
        })
        return env_config
    
class NoisyChangingTargetEnv(ChangingTargetEnv, NoisySingleQubitEnv):
    @classmethod
    def get_default_env_config(cls):
        config_dict = super().get_default_env_config()
        config_dict["observation_space_size"] = 35
        return config_dict
    
    def __init__(self, env_config):
        super().__init__(env_config)
        self.U_target_list = env_config["U_target_list"]
        self.target_generation_function = env_config["target_generation_function"]

    def set_target_gate(self):
        if len(self.U_target_list) == 0:
            U = self.target_generation_function().get_matrix()
        else:
            U = random.choice(self.U_target_list).get_matrix()
        self.U_target = self.unitary_to_superoperator(U)
        self.U_target_dm = U.copy()

    def set_initial_gate(self):
        self.U_initial_dm = self.target_generation_function().get_matrix()
        self.U_initial = self.unitary_to_superoperator(self.U_initial_dm.copy())

    def get_observation(self):
        U_diff = super().get_observation()
        normalized_detuning = [normalize(self.detuning, self.detuning_list)]
        normalized_relaxation_rates = [normalize(self.relaxation_rate[0], self.relaxation_rates_list[0]),
                                       normalize(self.relaxation_rate[1],
                                                 self.relaxation_rates_list[1])]  # could do list comprehension
        return np.append(normalized_relaxation_rates + normalized_detuning, U_diff)


    def return_env_config(self):
        env_config = super().return_env_config()
        env_config.update({"detuning_list": self.detuning_list,  # qubit detuning
                           "relaxation_rates_list": self.relaxation_rates_list,
                           "relaxation_ops": self.relaxation_ops,
                           "observation_space_size": 35,
                           })
        return env_config


