import gymnasium as gym
import numpy as np
import scipy.linalg as la
import cmath
import random
from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj, tensor
from qutip.operators import *
from qutip import cnot, cphase
#from relaqs.api.reward_functions import negative_matrix_difference_norm

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

class TwoQubitGateSynth(gym.Env):
    @classmethod
    def get_default_env_config(cls):
        return {
            "action_space_size": 7,
            "U_initial": II,  # staring with I
            "U_target": CZ,  # target for CZ
            "final_time": 30E-9, # in seconds
            "num_Haar_basis": 4,  # number of Haar basis (need to update for odd combinations)
            "steps_per_Haar": 2,  # steps per Haar basis per episode
            "delta": [[0],[0]],  # qubit detuning
            "save_data_every_step": 1,
            "verbose": True,
#            "relaxation_rates_list": [[1/60E-6/2/np.pi],[1/30E-6/2/np.pi],[1/66E-6/2/np.pi],[1/5E-6/2/np.pi]], # relaxation lists of list of floats to be sampled from when resetting environment.
            "relaxation_rates_list": [[0],[0],[0],[0]], # for now
            "relaxation_ops": [sigmam1,sigmam2,Qobj(Z1),Qobj(Z2)], #relaxation operator lists for T1 and T2, respectively
#            "observation_space_size": 35, # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate
            "observation_space_size": 2*256 + 1 + 4 + 2 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 4 for relaxation rate + 2 for detuning
        }

    #physics: https://journals.aps.org/prapplied/pdf/10.1103/PhysRevApplied.10.054062, eq(2)
    #parameters: https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.021058
    #30 ns duration, g1 = 72.5 MHz, g2 = 71.5 MHz, g12 = 5 MHz
    #T1 = 60 us, 30 us
    #T2* = 66 us, 5 us

    def hamiltonian(self, delta1, delta2, alpha1, alpha2, twoQubitDetuning, gamma_magnitude1, gamma_phase1, gamma_magnitude2, gamma_phase2, g1 = 72.5E6, g2 = 71.5E6, g12 = 5E6):
        selfEnergyTerms = (delta1 + alpha1) * Z1 + (delta2 + alpha2) * Z2
        Qubit1ControlTerms = gamma_magnitude1 * (np.cos(gamma_phase1) * X1 + np.sin(gamma_phase1) * Y1)
        Qubit2ControlTerms = gamma_magnitude2 * (np.cos(gamma_phase2) * X2 + np.sin(gamma_phase2) * Y2)

        #omega1 = delta1+alpha1, omega2 = delta2+alpha2, omegaC = alphaC
#        Delta1 = delta1+alpha1-alphaC
#        Delta2 = delta2+alpha2-alphaC
#        twoQubitDetuning = 1/((1/Delta1 + 1/Delta2)/2)
        
        g_eff = g1*g2/twoQubitDetuning + g12
        interactionEnergy = g_eff*exchangeOperator

        energyTotal = selfEnergyTerms + interactionEnergy + Qubit1ControlTerms + Qubit2ControlTerms


#        print("coupling: ", f"{g_eff:7.3f}")
#        print("Delta1: ", f"{Delta1:7.3f}")
#        print("Delta2: ", f"{Delta2:7.3f}")
#        print("twoQubitDetuning: ", f"{twoQubitDetuning:7.3f}")

        return energyTotal

    def __init__(self, env_config):
        self.final_time = env_config["final_time"]  # Final time for the gates
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(env_config["observation_space_size"],))  # propagation operator elements + fidelity + relaxation + detuning
        self.action_space = gym.spaces.Box(low=-1*np.ones(7), high=np.ones(7)) #alpha1, alpha2, alphaC, gamma_magnitude1, gamma_phase1, gamma_magnitude2, gamma_phase2
        self.delta = env_config["delta"]  # detuning
        self.detuning = [0, 0]
        self.detuning_update()
        self.U_target = self.unitary_to_superoperator(env_config["U_target"])
        self.U_initial = self.unitary_to_superoperator(env_config["U_initial"])
        self.num_Haar_basis = env_config["num_Haar_basis"]
        self.steps_per_Haar = env_config["steps_per_Haar"]
        self.verbose = env_config["verbose"]
        self.relaxation_rates_list = env_config["relaxation_rates_list"]
        self.relaxation_ops = env_config["relaxation_ops"]
        self.relaxation_rate = self.get_relaxation_rate()
        self.current_Haar_num = 1  # starting with 1
        self.current_step_per_Haar = 1
        self.H_array = []  # saving all H's with Haar wavelet to be multiplied
        self.H_tot = []  # Haar wavelet multipied H summed up for each time bin
        self.L_array = []  # Liouvillian for each time bin
        self.U_array = []  # propagation operators for each time bin
        self.U = self.U_initial.copy()  # multiplied propagtion operators
        self.state = self.unitary_to_observation(self.U_initial)  # starting observation space
        self.prev_fidelity = 0  # previous step' fidelity for rewarding
        self.alpha_max = 4*np.pi/self.final_time
        #self.alpha_max = 0
        #self.alphaC_mod_max = 1.5E9  ## see https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.021058
        #self.alphaC_mod_max = 0.005E9  ## see https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.021058
        #self.alphaC0 = 1.0367E9 # coupler center frequency : 5.2GHz, qubit 1 center frequency: 4.16 GHz
        #self.alphaC0 = 0.01E9 # coupler center frequency : 5.2GHz, qubit 1 center frequency: 4.16 GHz        
        self.Delta0 = 100E6 
        self.Delta_mod_max = 25E6 
        self.gamma_phase_max = 1.1675 * np.pi
        self.gamma_magnitude_max = 1.8 * np.pi / self.final_time / self.steps_per_Haar
        self.transition_history = []
        self.episode_id = 0

    def detuning_update(self):
        # Random detuning selection
        if len(self.delta[0])==1:
            detuning1 = self.delta[0][0]
        else:
            detuning1 = random.sample(self.delta[0],k=1)[0]
            
        # Random detuning selection
        if len(self.delta[1])==1:
            detuning2 = self.delta[0][0]
        else:
            detuning2 = random.sample(self.delta[1],k=1)[0]

        self.detuning = [detuning1, detuning2]
        
        

    def unitary_to_superoperator(self, U):
        return (spre(Qobj(U)) * spost(Qobj(U))).data.toarray()

    def get_relaxation_rate(self):
        relaxation_size = len(self.relaxation_ops)      #get number of relaxation ops
        
        sampled_rate_list = []
        for ii in range(relaxation_size):
            sampled_rate_list.append(random.sample(self.relaxation_rates_list[ii],k=1)[0])

        return sampled_rate_list
            
    def get_observation(self):
        normalizedDetuning = [(self.detuning[0] - min(self.delta[0])+1E-15)/(max(self.delta[0])-min(self.delta[0])+1E-15), (self.detuning[1] - min(self.delta[1])+1E-15)/(max(self.delta[1])-min(self.delta[1])+1E-15)]
        return np.append([self.compute_fidelity()]+[x//6283185 for x in self.relaxation_rate]+normalizedDetuning, self.unitary_to_observation(self.U)) #6283185 assuming 500 nanosecond relaxation is max
    
    def compute_fidelity(self):
        U_target_dagger = self.U_target.conjugate().transpose()
        F = float(np.abs(np.trace(U_target_dagger @ self.U))) / (self.U.shape[0])
        return F

    def unitary_to_observation(self, U):
        return (
            np.array(
                [(abs(x), (cmath.phase(x) / 2 / np.pi + 1) / 2) for x in U.flatten()], 
                dtype=np.float64,
                )
            .squeeze()
            .reshape(-1)  # cmath phase gives -2pi to 2pi (?)
        )

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
        self.relaxation_rate = self.get_relaxation_rate()
        self.detuning = 0
        self.detuning_update()
        starting_observeration = self.get_observation()
        info = {}
        self.episode_id += 1
        return starting_observeration, info

    def step(self, action):
        num_time_bins = 2 ** (self.current_Haar_num - 1) # Haar number decides the number of time bins

        ### action space setting
        alpha1 = self.alpha_max * action[0] 
        alpha2 = self.alpha_max * action[1] 
        #alphaC = self.alphaC0 + self.alphaC_mod_max * action[2] 
        Delta = self.Delta0 + self.Delta_mod_max * action[2]

        # gamma is the complex amplitude of the control field
        gamma_magnitude1 = self.gamma_magnitude_max / 2 * (action[3] + 1)
        gamma_magnitude2 = self.gamma_magnitude_max / 2 * (action[4] + 1)

        gamma_phase1 = self.gamma_phase_max * action[5] 
        gamma_phase2 = self.gamma_phase_max * action[6]

        # Set noise opertors
        jump_ops = []
        for ii in range(len(self.relaxation_ops)):
            jump_ops.append(np.sqrt(self.relaxation_rate[ii]) * self.relaxation_ops[ii])

        # Hamiltonian with controls
        H = self.hamiltonian(self.delta[0][0], self.delta[1][0], alpha1, alpha2, Delta, gamma_magnitude1, gamma_phase1, gamma_magnitude2, gamma_phase2)
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
        self.U = np.eye(16)  # identity

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot[jj]), jump_ops, data_only=False, chi=None)).data.toarray()  # Liouvillian calc
            self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at

        # Reward and fidelity calculation
        fidelity = self.compute_fidelity()
        reward = (-5 * np.log10(1.0 - fidelity) + np.log10(1.0 - self.prev_fidelity)) + (5 * fidelity - self.prev_fidelity)
        self.prev_fidelity = fidelity

        self.state = self.get_observation()


        # if self.verbose is True:
        #     print(
        #         "F: ", f"{fidelity:7.3f}",
        #         "R: ", f"{reward:7.3f}",
        #     )

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
