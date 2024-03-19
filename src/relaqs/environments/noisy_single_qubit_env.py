""" Noisy single qubit gate synthesis environment using Haar basis. """
import random
import numpy as np
import scipy.linalg as la
from qutip import Qobj
from qutip.superoperator import liouvillian, spre, spost
from qutip.operators import sigmam
from relaqs.environments.single_qubit_env import SingleQubitEnv
from relaqs.api import gates

I = gates.I().get_matrix()
X = gates.X().get_matrix()
Y = gates.Y().get_matrix()
Z = gates.Z().get_matrix()

class NoisySingleQubitEnv(SingleQubitEnv):
    @classmethod
    def get_default_env_config(cls):
        env_config = super().get_default_env_config()
        env_config.update({"detuning_list": [0],  # qubit detuning
            "relaxation_rates_list": [[314159]], # relaxation lists of list of floats to be sampled from when resetting environment. (10 usec)
            "relaxation_ops": [sigmam()], #relaxation operator lists for T1 and T2, respectively
            "observation_space_size": 2*16 + 1 + 1 + 1}) # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 1 for relaxation rate + 1 for detuning})
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

    def unitary_to_superoperator(self, U):
        return (spre(Qobj(U)) * spost(Qobj(U))).data.toarray()

    def get_relaxation_rate(self):
        relaxation_size = len(self.relaxation_ops) # get number of relaxation ops
        
        sampled_rate_list = []
        for ii in range(relaxation_size):
            sampled_rate_list.append(random.sample(self.relaxation_rates_list[ii],k=1)[0])

        return sampled_rate_list
            
    def get_observation(self):
        normalizedDetuning = [(self.detuning - min(self.detuning_list)+1E-15)/(max(self.detuning_list)-min(self.detuning_list)+1E-15)]
        return np.append([self.compute_fidelity()]+[x//6283185 for x in self.relaxation_rate]+normalizedDetuning, self.unitary_to_observation(self.U)) #6283185 assuming 500 nanosecond relaxation is max
    
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

        # printing on the command line for quick viewing
        if self.verbose is True:
            print(
                "Step: ", f"{self.current_step_per_Haar}" + " episode id :" + f"{self.episode_id}",
                "Relaxation rates:")
            for rate in self.relaxation_rate:
                print(f"{rate:7.6f}")
            print(
                "F: ", f"{fidelity:7.3f}",
                "R: ", f"{reward:7.3f}",
                "amp: " f"{action[0]:7.3f}",
                "phase: " f"{action[1]:7.3f}",
            )

        self.update_transition_history(fidelity, reward, action)

        truncated, terminated = self.is_episode_over(fidelity)

        self.Haar_update()

        info = {}
        return (self.state, reward, terminated, truncated, info)
