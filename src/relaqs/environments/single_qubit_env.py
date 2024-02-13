""" Single qubit gate synthesis environment using Haar basis. """
import cmath
import gymnasium as gym
import numpy as np
import scipy.linalg as la

from relaqs.api import gates

I = gates.I().get_matrix()
X = gates.X().get_matrix()
Y = gates.Y().get_matrix()
Z = gates.Z().get_matrix()

class SingleQubitEnv(gym.Env):
    @classmethod
    def get_default_env_config(cls):
        return {
            "action_space_size": 3,
            "U_initial": I,
            "U_target": X,  
            "final_time": 35.5556E-9, # in seconds
            "num_Haar_basis": 1,
            "steps_per_Haar": 2,  # steps per Haar basis per episode
            "delta": 0,
            "verbose": True,
            "observation_space_size": 9,  # 1 (fidelity) + 8 (flattened unitary)
        }
    def __init__(self, env_config):
        self.final_time = env_config["final_time"]  # Final time for the gates
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(env_config["observation_space_size"],))
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))
        self.delta = env_config["delta"] # detuning
        self.U_target = env_config["U_target"]
        self.U_initial = env_config["U_initial"] # future todo, can make random initial state
        self.U = env_config["U_initial"].copy()
        self.num_Haar_basis = env_config["num_Haar_basis"]
        self.steps_per_Haar = env_config["steps_per_Haar"]
        self.verbose = env_config["verbose"]
        self.current_Haar_num = 1
        self.current_step_per_Haar = 1
        self.H_array = []
        self.H_tot = []
        self.U_array = []
        self.state = self.unitary_to_observation(self.U)
        self.prev_fidelity = 0
        self.gamma_phase_max = 1.1675 * np.pi
        self.gamma_magnitude_max = 1.8 * np.pi / self.final_time / self.steps_per_Haar
        self.gamma_detuning_max = 0.05E9 # detuning of the control pulse in Hz
        self.transition_history = []
        self.episode_id = 0

    def unitary_to_observation(self, U):
        return (
            np.array(
                [(abs(x), (cmath.phase(x) / np.pi + 1) / 2) for x in U.flatten()],
                dtype=np.float64,
            )
            .squeeze()
            .reshape(-1)
        )

    def get_observation(self):
        return np.append([self.compute_fidelity()], self.unitary_to_observation(self.U))
    
    def compute_fidelity(self):
        U_target_dagger = self.U_target.conjugate().transpose()
        return float(np.abs(np.trace(U_target_dagger @ self.U))) / (self.U.shape[0])
    
    def compute_reward(self, fidelity):
        return (-3 * np.log10(1.0 - fidelity) + np.log10(1.0 - self.prev_fidelity)) + (3 * fidelity - self.prev_fidelity)
        
    def hamiltonian(self, delta, alpha, gamma_magnitude, gamma_phase):
        return (delta + alpha) * Z + gamma_magnitude * (np.cos(gamma_phase) * X + np.sin(gamma_phase) * Y)

    def reset(self, *, seed=None, options=None):
        self.U = self.U_initial.copy()
        starting_observeration = self.get_observation()
        self.state = self.get_observation()
        self.current_Haar_num = 1
        self.current_step_per_Haar = 1
        self.H_array = []
        self.H_tot = []
        self.U_array = []
        self.prev_fidelity = 0
        info = {}
        self.episode_id += 1
        return starting_observeration, info
    
    def hamiltonian_update(self, alpha, gamma_magnitude, gamma_phase, num_time_bins):
        H = self.hamiltonian(self.delta, alpha, gamma_magnitude, gamma_phase)
        self.H_array.append(H)

        self.H_tot = []

        for ii, H_elem in enumerate(self.H_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(ii / self.steps_per_Haar) # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1)))
                if ii > 0:
                    self.H_tot[jj] += factor * H_elem
                else:
                    self.H_tot.append(factor * H_elem)

        self.U = self.U_initial.copy()

    def is_episode_over(self, fidelity):
        truncated = False
        terminated = False
        if fidelity >= 1:
            truncated = True  # truncated when target fidelity reached
        elif (self.current_Haar_num >= self.num_Haar_basis) and (self.current_step_per_Haar >= self.steps_per_Haar):  # terminate when all Haar is tested
            terminated = True
        else:
            terminated = False
        return truncated, terminated

    def Haar_update(self):
        if (self.current_step_per_Haar == self.steps_per_Haar):  # For each Haar basis, if all trial steps ends, them move to next haar wavelet
            self.current_Haar_num += 1
            self.current_step_per_Haar = 1
        else:
            self.current_step_per_Haar += 1

    def parse_actions(self, action):
        gamma_magnitude = self.gamma_magnitude_max / 2 * (action[0] + 1)
        gamma_phase = self.gamma_phase_max * action[1]
        alpha = self.gamma_detuning_max * action[2]
        return gamma_magnitude, gamma_phase, alpha
    
    def update_transition_history(self, fidelity, reward, action):
        self.transition_history.append([fidelity, reward, action, self.U, self.episode_id])

    def step(self, action):
        num_time_bins = 2 ** (self.current_Haar_num - 1) # Haar number decides the number of time bins

        # Get actions
        gamma_magnitude, gamma_phase, alpha = self.parse_actions(action)

        self.hamiltonian_update(alpha, gamma_magnitude, gamma_phase, num_time_bins)

        # U update
        for jj in range(0, num_time_bins):
            Ut = la.expm(-1j * self.final_time / num_time_bins * self.H_tot[jj])
            self.U = Ut @ self.U
        self.U_array.append(self.U)

        # Get reward (fidelity)
        fidelity = self.compute_fidelity()
        reward = self.compute_reward(fidelity)
        self.prev_fidelity = fidelity

        self.state = self.get_observation()

        # printing on the command line for quick viewing
        if self.verbose is True:
            print(
                "Step: ", f"{self.current_step_per_Haar}",
                "F: ", f"{fidelity:7.3f}",
                "R: ", f"{reward:7.3f}",
                "amp: " f"{action[0]:7.3f}",
                "phase: " f"{action[1]:7.3f}",
                "detuning: " f"{action[2]:7.3f}"
            )

        self.update_transition_history(fidelity, reward, action)
        
        self.Haar_update()

        truncated, terminated = self.is_episode_over(fidelity)

        info = {}
        return (self.state, reward, terminated, truncated, info)
