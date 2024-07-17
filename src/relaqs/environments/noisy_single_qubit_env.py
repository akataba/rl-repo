""" Noisy single qubit gate synthesis environment using Haar basis. """
import random
import numpy as np
import scipy.linalg as la
from qutip import Qobj
from qutip.superoperator import liouvillian
from qutip.operators import sigmam, sigmaz
from relaqs.environments.single_qubit_env import SingleQubitEnv
from relaqs.api import gates
from relaqs.api.utils import sample_noise_parameters

I = gates.I().get_matrix()
X = gates.X().get_matrix()
Y = gates.Y().get_matrix()
Z = gates.Z().get_matrix()

class NoisySingleQubitEnv(SingleQubitEnv):
    @classmethod
    def get_default_env_config(cls):
        env_config = super().get_default_env_config()
        t1_list, t2_list, detuning_list = sample_noise_parameters()
        env_config.update({"detuning_list": detuning_list,  # qubit detuning
            "relaxation_rates_list": [t1_list, t2_list], # relaxation lists of list of floats to be sampled from when resetting environment. (10 usec)
            "relaxation_ops": [sigmam(), sigmaz()], #relaxation operator lists for T1 and T2, respectively
            "observation_space_size": 2*16 + 1 + 2 + 1}) # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rates + 1 for detuning})
        return env_config

    def __init__(self, env_config):
        super().__init__(env_config)
        self.detuning_list = env_config["detuning_list"]
        self.detuning_update()
        self.U_target = self.unitary_to_superoperator(env_config["U_target"])
        self.U_initial = self.unitary_to_superoperator(env_config["U_initial"])
        self.relaxation_rates_list = env_config["relaxation_rates_list"]
        self.relaxation_ops = env_config["relaxation_ops"]
        self.relaxation_rate = self.get_relaxation_rate()
        self.U = self.U_initial.copy()  # multiplied propagtion operators
        self.state = self.unitary_to_observation(self.U_initial)  # starting observation space

    def detuning_update(self):
        # Random detuning selection
        if len(self.detuning_list)==1:
            self.detuning = self.detuning_list[0]
        else:
            self.detuning = random.sample(self.detuning_list, k=1)[0]
            print("detuning: ", f"{self.detuning}")

    @classmethod
    def unitary_to_superoperator(self, U):
        return np.kron(U.conj(), U)

    def get_relaxation_rate(self):
        relaxation_size = len(self.relaxation_ops) # get number of relaxation ops
        
        sampled_rate_list = []
        for ii in range(relaxation_size):
            sampled_rate_list.append(random.sample(self.relaxation_rates_list[ii],k=1)[0])

        return sampled_rate_list
            
    def get_observation(self):
        normalized_detuning = [(self.detuning - min(self.detuning_list) + 1E-15) / (max(self.detuning_list) - min(self.detuning_list) + 1E-15)]
        return np.append([self.compute_fidelity()] +
                         [x // 6283185 for x in self.relaxation_rate] +
                         normalized_detuning,
                         self.unitary_to_observation(self.U)) #6283185 assuming 500 nanosecond relaxation is max
    
    def hamiltonian(self, detuning, alpha, gamma_magnitude, gamma_phase):
        return (detuning + alpha)*Z + gamma_magnitude*(np.cos(gamma_phase)*X + np.sin(gamma_phase)*Y)

    def reset(self, *, seed=None, options=None):
        super().reset()
        self.state = self.get_observation()
        self.relaxation_rate = self.get_relaxation_rate()
        self.detuning_update()
        starting_observeration = self.get_observation()
        info = {}
        return starting_observeration, info
    
    def operator_update(self, num_time_bins):
        # Set noise opertors
        jump_ops = []
        for ii in range(len(self.relaxation_ops)):
            jump_ops.append(np.sqrt(self.relaxation_rate[ii]) * self.relaxation_ops[ii])

        self.U = self.U_initial.copy()
        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot[jj]), jump_ops, data_only=False, chi=None)).data.toarray()  # Liouvillian calc
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at

    def get_info(self, fidelity, reward, action, truncated, terminated):
        info_string = super().get_info(fidelity, reward, action, truncated, terminated)
        info_string += f"Relaxation rate: {self.relaxation_rate}"
        return info_string

    def step(self, action):
        num_time_bins = 2 ** (self.current_Haar_num - 1) # Haar number decides the number of time bins

        # gamma is the complex amplitude of the control field
        gamma_magnitude, gamma_phase, alpha = self.parse_actions(action)

        self.hamiltonian_update(num_time_bins, self.detuning, alpha, gamma_magnitude, gamma_phase)

        self.operator_update(num_time_bins)

        # Reward and fidelity calculation
        fidelity = self.compute_fidelity()
        reward = self.compute_reward(fidelity)
        self.prev_fidelity = fidelity

        self.state = self.get_observation()

        self.update_transition_history(fidelity, reward, action)

        truncated, terminated = self.is_episode_over(fidelity)

        if self.verbose is True:
            print(self.get_info(fidelity, reward, action, truncated, terminated))

        self.Haar_update()

        info = {}
        return (self.state, reward, terminated, truncated, info)
