""" Noisy single qubit gate synthesis environment using Haar basis. """
import gymnasium as gym
import numpy as np
import scipy.linalg as la
import cmath
import random
from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj, tensor
from qutip.operators import *
from qutip import cnot, cphase

from relaqs.environments.single_qubit_env import GateSynthEnvRLlibHaar

sig_p = np.array([[0, 1], [0, 0]])
sig_m = np.array([[0, 0], [1, 0]])
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.array([[1, 0], [0, 1]])
Y = np.array([[0, 1j], [-1j, 0]])

#two-qubit single qubit gates
II = tensor(Qobj(I),Qobj(I)).data.toarray()
X1 = tensor(Qobj(X),Qobj(I)).data.toarray()
X2 = tensor(Qobj(I),Qobj(X)).data.toarray()
Y1 = tensor(Qobj(Y),Qobj(I)).data.toarray()
Y2 = tensor(Qobj(I),Qobj(Y)).data.toarray()
Z1 = tensor(Qobj(Z),Qobj(I)).data.toarray()
Z2 = tensor(Qobj(I),Qobj(Z)).data.toarray()

sig_p1 = tensor(Qobj(sig_p),Qobj(I)).data.toarray()
sig_p2 = tensor(Qobj(I),Qobj(sig_p)).data.toarray()
sig_m1 = tensor(Qobj(sig_m),Qobj(I)).data.toarray()
sig_m2 = tensor(Qobj(I),Qobj(sig_m)).data.toarray()
sigmap1 = Qobj(sig_p1)
sigmap2 = Qobj(sig_p2)
sigmam1 = Qobj(sig_m1)
sigmam2 = Qobj(sig_m2)

#two-qubit gate basis
XX = tensor(Qobj(X),Qobj(X)).data.toarray()
YY = tensor(Qobj(Y),Qobj(Y)).data.toarray()
ZZ = tensor(Qobj(Z),Qobj(Z)).data.toarray()
exchangeOperator = tensor(Qobj(sig_p),Qobj(sig_m)).data.toarray() + tensor(Qobj(sig_m),Qobj(sig_p)).data.toarray()

CNOT = cnot().data.toarray()
CZ = cphase(np.pi).data.toarray()


class GateSynthEnvRLlibHaarNoisy(GateSynthEnvRLlibHaar):
    @classmethod
    def get_default_env_config(cls):
        env_config = super().get_default_env_config()
        env_config.update({"delta": [0],  # qubit detuning
            "relaxation_rates_list": [[314159]], # relaxation lists of list of floats to be sampled from when resetting environment. (10 usec)
            "relaxation_ops": [sigmam()], #relaxation operator lists for T1 and T2, respectively
            "observation_space_size": 2*16 + 1 + 1 + 1}) # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 1 for relaxation rate + 1 for detuning})
        return env_config

    def __init__(self, env_config):
        super().__init__(env_config)
        self.delta = env_config["delta"]
        #self.detuning = 0
        self.detuning_update()
        self.U_target = self.unitary_to_superoperator(env_config["U_target"])
        self.U_initial = self.unitary_to_superoperator(env_config["U_initial"])
        self.relaxation_rates_list = env_config["relaxation_rates_list"]
        self.relaxation_ops = env_config["relaxation_ops"]
        self.relaxation_rate = self.get_relaxation_rate()
        self.L_array = []  # Liouvillian for each time bin
        self.U = self.U_initial.copy()  # multiplied propagtion operators
        self.state = self.unitary_to_observation(self.U_initial)  # starting observation space

    def detuning_update(self):
        # Random detuning selection
        if len(self.delta)==1:
            self.detuning = self.delta[0]
        else:
            self.detuning = random.sample(self.delta,k=1)[0]
            print("detuning: ", f"{self.detuning}")

    def unitary_to_superoperator(self, U):
        return (spre(Qobj(U)) * spost(Qobj(U))).data.toarray()

    def get_relaxation_rate(self):
        relaxation_size = len(self.relaxation_ops)      #get number of relaxation ops
        
        sampled_rate_list = []
        for ii in range(relaxation_size):
            sampled_rate_list.append(random.sample(self.relaxation_rates_list[ii],k=1)[0])

        return sampled_rate_list
            
    def get_observation(self):
        normalizedDetuning = [(self.detuning - min(self.delta)+1E-15)/(max(self.delta)-min(self.delta)+1E-15)]
        return np.append([self.compute_fidelity()]+[x//6283185 for x in self.relaxation_rate]+normalizedDetuning, self.unitary_to_observation(self.U)) #6283185 assuming 500 nanosecond relaxation is max

    def reset(self, *, seed=None, options=None):
        self.U = self.U_initial.copy()
        self.state = self.get_observation()
        self.current_Haar_num = 1
        self.current_step_per_Haar = 1
        self.H_array = []
        self.H_tot = []
        self.L_array = []
        self.U_array = []
        self.prev_fidelity = 0
        self.episode_id += 1
        self.relaxation_rate = self.get_relaxation_rate()
        #self.detuning = 0
        self.detuning_update()
        starting_observeration = self.get_observation()
        info = {}
        return starting_observeration, info

    def step(self, action):
        num_time_bins = 2 ** (self.current_Haar_num - 1) # Haar number decides the number of time bins

        # gamma is the complex amplitude of the control field
        gamma_magnitude = self.gamma_magnitude_max / 2 * (action[0] + 1)
        gamma_phase = self.gamma_phase_max * action[1]
        alpha = self.gamma_detuning_max * action[2]

        # Set noise opertors
        jump_ops = []
        for ii in range(len(self.relaxation_ops)):
            jump_ops.append(np.sqrt(self.relaxation_rate[ii]) * self.relaxation_ops[ii])

        # Hamiltonian with controls
        H = self.hamiltonian(self.detuning, alpha, gamma_magnitude, gamma_phase)
        self.H_array.append(H)  # Array of Hs at each Haar wavelet

        # H_tot for adding Hs at each time bins
        self.H_tot = []

        for ii, H_elem in enumerate(self.H_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(ii / self.steps_per_Haar) # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1))) # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot.append(factor * H_elem)

        self.L = ([])  # at every step we calculate L again because minimal time bin changes
        self.U = np.eye(4)  # identity

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot[jj]), jump_ops, data_only=False, chi=None)).data.toarray()  # Liouvillian calc
            self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at

        # Reward and fidelity calculation
        fidelity = self.compute_fidelity()
        reward = (-3 * np.log10(1.0 - fidelity) + np.log10(1.0 - self.prev_fidelity)) + (3 * fidelity - self.prev_fidelity)
        #reward = negative_matrix_difference_norm(self.U_target, self.U)
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

        self.transition_history.append([fidelity, reward, action, self.U, self.episode_id])

        # Determine if episode is over
        truncated = False
        terminated = False
        if fidelity >= 1:
            truncated = True  # truncated when target fidelity reached
        elif (self.current_Haar_num >= self.num_Haar_basis) and (self.current_step_per_Haar >= self.steps_per_Haar):  # terminate when all Haar is tested
            terminated = True
        else:
            terminated = False

        if (self.current_step_per_Haar == self.steps_per_Haar):  # For each Haar basis, if all trial steps ends, them move to next haar wavelet
            self.current_Haar_num += 1
            self.current_step_per_Haar = 1
        else:
            self.current_step_per_Haar += 1

        info = {}
        return (self.state, reward, terminated, truncated, info)