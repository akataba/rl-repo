from typing import Tuple

import cmath
import random
from scipy.linalg import expm

from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj, tensor
from qutip.operators import *
from qutip import cnot, cphase

from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer

import gymnasium as gym
import numpy as np
import scipy.linalg as la

sig_p = np.array([[0, 1.], [0, 0]])
sig_m = np.array([[0, 0], [1., 0]])
X = np.array([[0, 1.], [1., 0]])
Z = np.array([[1., 0], [0, -1.]])
I = np.array([[1., 0], [0, 1.]])
Y = np.array([[0, -1.j], [1.j, 0]])

H = np.array([[1/np.sqrt(2),1/np.sqrt(2)],[1/np.sqrt(2),-1/np.sqrt(2)]])
S = np.array([[1.,0],[0,1.j]])
Sdagger = np.array([[1.,0],[0,-1.j]])

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
exchangeOperator1 = XX+YY
exchangeOperator2 = YY+ZZ
exchangeOperator3 = XX+ZZ

CNOT = cnot().data.toarray()
CZ = cphase(np.pi).data.toarray()


class NoisyTwoQubitEnv(gym.Env):
    @classmethod
    def get_default_env_config(cls):
        return {
            "action_space_size": 27,
            "U_initial": II,  # staring with I
            "U_target": CZ,  # target for CZ
            "final_time": 30E-9, # in seconds, total time is final_time * 5 because of single qubit + two_qubit + single_qubit + two_qubit + single_qubit
            "num_Haar_basis": 3,  # number of Haar basis (need to update for odd combinations)
            "steps_per_Haar": 1,  # steps per Haar basis per episode
            "delta": np.random.normal(0, np.pi/100/30E-9, size=(2, 100)).tolist(),  # qubit detuning
            # "delta": [[np.pi/60/30E-9],[-np.pi/85/30E-9]],  # qubit detuning
            # "delta": [[0],[0]],  # qubit detuning
            "save_data_every_step": 1,
            "verbose": True,
            # "relaxation_rates_list": [[1/60E-6/2/np.pi],[1/30E-6/2/np.pi],[1/66E-6/2/np.pi],[1/5E-6/2/np.pi]], # relaxation lists of list of floats to be sampled from when resetting environment.
            "relaxation_rates_list": [[0],[0],[0],[0]], # for now
            "relaxation_ops": [sigmam1,sigmam2,Qobj(Z1),Qobj(Z2)], #relaxation operator lists for T1 and T2, respectively
            # "observation_space_size": 2*256 + 1 + 4 + 2 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 4 for relaxation rate + 2 for detuning
            "observation_space_size": 2*16 + 1 + 4 + 2 # 2*16 = (complex number)*(target unitary matrix elements = 4)^2, + 1 for fidelity + 4 for relaxation rate + 2 for detuning
        }

    #physics: https://journals.aps.org/prapplied/pdf/10.1103/PhysRevApplied.10.054062, eq(2)
    #parameters: https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.021058
    #30 ns duration, g1 = 72.5 MHz, g2 = 71.5 MHz, g12 = 5 MHz
    #T1 = 60 us, 30 us
    #T2* = 66 us, 5 us

    def hamiltonian(self, delta1, delta2, alpha1, alpha2, g_eff, gamma_magnitude1, gamma_phase1, gamma_magnitude2, gamma_phase2, index = 1):
        selfEnergyTerms = (delta1 + alpha1) * Z1 + (delta2 + alpha2) * Z2
        Qubit1ControlTerms = gamma_magnitude1 * (np.cos(gamma_phase1) * X1 + np.sin(gamma_phase1) * Y1)
        Qubit2ControlTerms = gamma_magnitude2 * (np.cos(gamma_phase2) * X2 + np.sin(gamma_phase2) * Y2)
        
        if index ==1:
            interactionEnergy = g_eff*exchangeOperator1
        elif index ==2:
            interactionEnergy = g_eff*exchangeOperator2
        elif index ==3:
            interactionEnergy = g_eff*exchangeOperator3
        else:
            interactionEnergy = 0
            print("interaction kind not specified")

        energyTotal = selfEnergyTerms + interactionEnergy + Qubit1ControlTerms + Qubit2ControlTerms

        return energyTotal

    def __init__(self, env_config):
        self.final_time = env_config["final_time"]  # Final time for the gates
        self.PiFreq = np.pi / self.final_time
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(env_config["observation_space_size"],))  # propagation operator elements + fidelity + relaxation + detuning
        self.action_space = gym.spaces.Box(low=-0.1*np.ones(27), high=0.1*np.ones(27)) #alpha1, alpha2, alphaC, gamma_magnitude1, gamma_phase1, gamma_magnitude2, gamma_phase2
        self.delta = env_config["delta"]  # detuning
        self.detuning = [0, 0]
        self.detuning_update()
        self._U_target = self.unitary_to_superoperator(env_config["U_target"])
        self.unitary_U_target = env_config["U_target"]
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
        # self.state = self.unitary_to_observation(self.U_initial)  # starting observation space
        self.state = self.unitary_to_observation(self.unitary_U_target)  # starting observation space
        self.prev_fidelity = 0  # previous step' fidelity for rewarding
        self.alpha_max = self.PiFreq / 2
        self.g_eff_max = self.PiFreq / 2
        self.gamma_phase_max = np.pi
        self.gamma_magnitude_max = self.PiFreq / 2
        self.transition_history = []
        self.env_config = env_config
        self.initialActions = self.KakActionCalculation()

    def detuning_update(self):
        # Random detuning selection
        if len(self.delta[0])==1:
            detuning1 = self.delta[0][0]
        else:
            detuning1 = random.sample(self.delta[0],k=1)[0]
            
        # Random detuning selection
        if len(self.delta[1])==1:
            detuning2 = self.delta[1][0]
        else:
            detuning2 = random.sample(self.delta[1],k=1)[0]

        self.detuning = [detuning1, detuning2]
        
    def update_target_unitary(self, U):
        self._U_target = self.unitary_to_superoperator(U)
        self.unitary_U_target = U
        self.initialActions = self.KakActionCalculation()    

    def unitary_to_superoperator(self, U):
        return (spre(Qobj(U)) * spost(Qobj(U.conjugate().transpose()))).data.toarray()

    def get_relaxation_rate(self):
        relaxation_size = len(self.relaxation_ops)      #get number of relaxation ops
        
        sampled_rate_list = []
        for ii in range(relaxation_size):
            sampled_rate_list.append(random.sample(self.relaxation_rates_list[ii],k=1)[0])

        return sampled_rate_list
            
    def get_observation(self):
        normalizedDetuning = [(self.detuning[0] - min(self.delta[0])+1E-15)/(max(self.delta[0])-min(self.delta[0])+1E-15), (self.detuning[1] - min(self.delta[1])+1E-15)/(max(self.delta[1])-min(self.delta[1])+1E-15)]
        # return np.append([self.compute_fidelity()]+[x//6283185 for x in self.relaxation_rate]+normalizedDetuning, self.unitary_to_observation(self.U)) #6283185 assuming 500 nanosecond relaxation is max
        return np.append([self.compute_fidelity()]+[x//6283185 for x in self.relaxation_rate]+normalizedDetuning, self.unitary_to_observation(self.unitary_U_target)) #6283185 assuming 500 nanosecond relaxation is max
    
    def compute_fidelity(self):
        U_target_dagger = self.unitary_to_superoperator(self.unitary_U_target.conjugate().transpose())
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
        self.initialActions = self.KakActionCalculation()        
        self.state = self.get_observation()
        self.current_Haar_num = 1
        self.current_step_per_Haar = 1
        self.H1_1_array = []
        self.H2_1_array = []
        self.H1_2_array = []
        self.H2_2_array = []
        self.H1_3_array = []
        self.H2_3_array = []        
        self.H1_4_array = []
        
        self.H_tot1_1 = []
        self.H_tot2_1 = []
        self.H_tot1_2 = []
        self.H_tot2_2 = []
        self.H_tot1_3 = []        
        self.H_tot2_3 = []
        self.H_tot1_4 = []        
        
        self.L_array = []
        self.U_array = []
        self.prev_fidelity = 0
        self.relaxation_rate = self.get_relaxation_rate()
        self.detuning = 0
        self.detuning_update()
        starting_observeration = self.get_observation()
        info = {}
        return starting_observeration, info

    def step(self, action):
        num_time_bins = 2 ** (self.current_Haar_num - 1)
        self.initialActions = self.KakActionCalculation()
        
        ### First single qubit gate
        
        if self.current_Haar_num==1:
            alpha1_1 = self.alpha_max * (action[0] + self.initialActions[0])
            alpha2_1 = self.alpha_max * (action[1] + self.initialActions[1])

            gamma_magnitude1_1 = self.gamma_magnitude_max * (action[2] + self.initialActions[2])
            gamma_magnitude2_1 = self.gamma_magnitude_max * (action[3] + self.initialActions[3])

            gamma_phase1_1 = self.gamma_phase_max * (action[4] + self.initialActions[4])
            gamma_phase2_1 = self.gamma_phase_max * (action[5] + self.initialActions[5])
        elif self.current_Haar_num==2:
            alpha1_1 = - self.alpha_max * (action[0] + self.initialActions[0])
            alpha2_1 = - self.alpha_max * (action[1] + self.initialActions[1])

            gamma_magnitude1_1 = self.gamma_magnitude_max * (action[2] + self.initialActions[2])
            gamma_magnitude2_1 = self.gamma_magnitude_max * (action[3] + self.initialActions[3])

            gamma_phase1_1 = self.gamma_phase_max * (action[4] + self.initialActions[4])
            gamma_phase2_1 = self.gamma_phase_max * (action[5] + self.initialActions[5])
        else:
            alpha1_1 = self.alpha_max * (action[0])
            alpha2_1 = self.alpha_max * (action[1])

            gamma_magnitude1_1 = self.gamma_magnitude_max * (action[2])
            gamma_magnitude2_1 = self.gamma_magnitude_max * (action[3])

            gamma_phase1_1 = self.gamma_phase_max * (action[4])
            gamma_phase2_1 = self.gamma_phase_max * (action[5])            

        ### First two qubit gate
        
        if self.current_Haar_num==1:
            g_eff1 = self.g_eff_max * (action[6]+self.initialActions[6])
        else:
            g_eff1 = self.g_eff_max * (action[6])

        ### Second Single qubit gate
        if self.current_Haar_num==1:
            alpha1_2 = self.alpha_max * (action[7] + self.initialActions[7])
            alpha2_2 = self.alpha_max * (action[8] + self.initialActions[8])

            gamma_magnitude1_2 = self.gamma_magnitude_max * (action[9] + self.initialActions[9])
            gamma_magnitude2_2 = self.gamma_magnitude_max * (action[10] + self.initialActions[10])

            gamma_phase1_2 = self.gamma_phase_max * (action[11] + self.initialActions[11])
            gamma_phase2_2 = self.gamma_phase_max * (action[12] + self.initialActions[12])
        elif self.current_Haar_num==2:
            alpha1_2 = - self.alpha_max * (action[7] + self.initialActions[7])
            alpha2_2 = - self.alpha_max * (action[8] + self.initialActions[8])

            gamma_magnitude1_2 = self.gamma_magnitude_max * (action[9] + self.initialActions[9])
            gamma_magnitude2_2 = self.gamma_magnitude_max * (action[10] + self.initialActions[10])

            gamma_phase1_2 = self.gamma_phase_max * (action[11] + self.initialActions[11])
            gamma_phase2_2 = self.gamma_phase_max * (action[12] + self.initialActions[12])
        else:
            alpha1_2 = self.alpha_max * (action[7])
            alpha2_2 = self.alpha_max * (action[8])

            gamma_magnitude1_2 = self.gamma_magnitude_max * (action[9])
            gamma_magnitude2_2 = self.gamma_magnitude_max * (action[10])

            gamma_phase1_2 = self.gamma_phase_max * (action[11])
            gamma_phase2_2 = self.gamma_phase_max * (action[12])
            
        ### second two qubit gate
        if self.current_Haar_num==1:
            g_eff2 = self.g_eff_max * (action[13]+self.initialActions[13])
        else:
            g_eff2 = self.g_eff_max * (action[13])

        ### Third Single qubit gate
        if self.current_Haar_num==1:        
            alpha1_3 = self.alpha_max * (action[14] + self.initialActions[14])
            alpha2_3 = self.alpha_max * (action[15] + self.initialActions[15])

            gamma_magnitude1_3 = self.gamma_magnitude_max * (action[16] + self.initialActions[16])
            gamma_magnitude2_3 = self.gamma_magnitude_max * (action[17] + self.initialActions[17])

            gamma_phase1_3 = self.gamma_phase_max * (action[18] + self.initialActions[18]) 
            gamma_phase2_3 = self.gamma_phase_max * (action[19] + self.initialActions[19])
        elif self.current_Haar_num==2:
            alpha1_3 = - self.alpha_max * (action[14] + self.initialActions[14])
            alpha2_3 = - self.alpha_max * (action[15] + self.initialActions[15])

            gamma_magnitude1_3 = self.gamma_magnitude_max * (action[16] + self.initialActions[16])
            gamma_magnitude2_3 = self.gamma_magnitude_max * (action[17] + self.initialActions[17])

            gamma_phase1_3 = self.gamma_phase_max * (action[18] + self.initialActions[18]) 
            gamma_phase2_3 = self.gamma_phase_max * (action[19] + self.initialActions[19])
        else:
            alpha1_3 = self.alpha_max * (action[14])
            alpha2_3 = self.alpha_max * (action[15])

            gamma_magnitude1_3 = self.gamma_magnitude_max * (action[16])
            gamma_magnitude2_3 = self.gamma_magnitude_max * (action[17])

            gamma_phase1_3 = self.gamma_phase_max * (action[18])
            gamma_phase2_3 = self.gamma_phase_max * (action[19])

        ### third two qubit gate
        if self.current_Haar_num==1:
            g_eff3 = self.g_eff_max * (action[20]+self.initialActions[20])
        else:
            g_eff3 = self.g_eff_max * (action[20])
        
        ### Fourth Single qubit gate
        if self.current_Haar_num==1:        
            alpha1_4 = self.alpha_max * (action[21] + self.initialActions[21])
            alpha2_4 = self.alpha_max * (action[22] + self.initialActions[22])

            gamma_magnitude1_4 = self.gamma_magnitude_max * (action[23] + self.initialActions[23])
            gamma_magnitude2_4 = self.gamma_magnitude_max * (action[24] + self.initialActions[24])

            gamma_phase1_4 = self.gamma_phase_max * (action[25] + self.initialActions[25])
            gamma_phase2_4 = self.gamma_phase_max * (action[26] + self.initialActions[26])
        elif self.current_Haar_num==2:
            alpha1_4 = - self.alpha_max * (action[21] + self.initialActions[21])
            alpha2_4 = - self.alpha_max * (action[22] + self.initialActions[22])

            gamma_magnitude1_4 = self.gamma_magnitude_max * (action[23] + self.initialActions[23])
            gamma_magnitude2_4 = self.gamma_magnitude_max * (action[24] + self.initialActions[24])

            gamma_phase1_4 = self.gamma_phase_max * (action[25] + self.initialActions[25])
            gamma_phase2_4 = self.gamma_phase_max * (action[26] + self.initialActions[26])
        else:            
            alpha1_4 = self.alpha_max * (action[21])
            alpha2_4 = self.alpha_max * (action[22])

            gamma_magnitude1_4 = self.gamma_magnitude_max * (action[23])
            gamma_magnitude2_4 = self.gamma_magnitude_max * (action[24])

            gamma_phase1_4 = self.gamma_phase_max * (action[25])
            gamma_phase2_4 = self.gamma_phase_max * (action[26])

        # Set noise opertors
        jump_ops = []
        for ii in range(len(self.relaxation_ops)):
            jump_ops.append(np.sqrt(self.relaxation_rate[ii]) * self.relaxation_ops[ii])

        # Hamiltonian with controls
        H2_1 = self.hamiltonian(self.detuning[0], self.detuning[1], 0, 0, g_eff1, 0, 0, 0, 0, index = 1)
        H2_2 = self.hamiltonian(self.detuning[0], self.detuning[1], 0, 0, g_eff2, 0, 0, 0, 0, index = 1)
        H2_3 = self.hamiltonian(self.detuning[0], self.detuning[1], 0, 0, g_eff3, 0, 0, 0, 0, index = 1)
        
        H1_1 = self.hamiltonian(self.detuning[0], self.detuning[1], alpha1_1, alpha2_1, 0, gamma_magnitude1_1, gamma_phase1_1, gamma_magnitude2_1, gamma_phase2_1)
        H1_2 = self.hamiltonian(self.detuning[0], self.detuning[1], alpha1_2, alpha2_2, 0, gamma_magnitude1_2, gamma_phase1_2, gamma_magnitude2_2, gamma_phase2_2)
        H1_3 = self.hamiltonian(self.detuning[0], self.detuning[1], alpha1_3, alpha2_3, 0, gamma_magnitude1_3, gamma_phase1_3, gamma_magnitude2_3, gamma_phase2_3)
        H1_4 = self.hamiltonian(self.detuning[0], self.detuning[1], alpha1_4, alpha2_4, 0, gamma_magnitude1_4, gamma_phase1_4, gamma_magnitude2_4, gamma_phase2_4)

        
        self.H2_1_array.append(H2_1)  # Array of Hs at each Haar wavelet
        self.H2_2_array.append(H2_2)  # Array of Hs at each Haar wavelet
        self.H2_3_array.append(H2_3)  # Array of Hs at each Haar wavelet

        self.H1_1_array.append(H1_1)  # Array of Hs at each Haar wavelet
        self.H1_2_array.append(H1_2)  # Array of Hs at each Haar wavelet
        self.H1_3_array.append(H1_3)  # Array of Hs at each Haar wavelet
        self.H1_4_array.append(H1_4)  # Array of Hs at each Haar wavelet
        
        # H_tot for adding Hs at each time bins
        self.H_tot2_1 = []
        self.H_tot2_2 = []
        self.H_tot2_3 = []

        self.H_tot1_1 = []
        self.H_tot1_2 = []
        self.H_tot1_3 = []
        self.H_tot1_4 = []
        
        for ii, H_elem in enumerate(self.H1_1_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(ii / self.steps_per_Haar) # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1))) # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot1_1[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot1_1.append(factor * H_elem)          

        for ii, H_elem in enumerate(self.H2_1_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(ii / self.steps_per_Haar) # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1))) # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot2_1[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot2_1.append(factor * H_elem)

        for ii, H_elem in enumerate(self.H1_2_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(ii / self.steps_per_Haar) # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1))) # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot1_2[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot1_2.append(factor * H_elem)

        for ii, H_elem in enumerate(self.H2_2_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(ii / self.steps_per_Haar) # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1))) # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot2_2[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot2_2.append(factor * H_elem)

        for ii, H_elem in enumerate(self.H1_3_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(ii / self.steps_per_Haar) # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1))) # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot1_3[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot1_3.append(factor * H_elem)

        for ii, H_elem in enumerate(self.H2_3_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(ii / self.steps_per_Haar) # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1))) # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot2_3[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot2_3.append(factor * H_elem)

        for ii, H_elem in enumerate(self.H1_4_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(ii / self.steps_per_Haar) # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num - 1))) # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:
                    self.H_tot1_4[jj] += factor * H_elem
                else:  # Because H_tot[jj] does not exist
                    self.H_tot1_4.append(factor * H_elem)

        self.L = ([])  # at every step we calculate L again because minimal time bin changes
        self.U = np.eye(16)  # identity
        self.unitary_U = np.eye(4)

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot1_1[jj]), jump_ops, data_only=False, chi=None)).data.toarray()  # Liouvillian calc
            self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            unitary_Ut = la.expm(-1j * self.H_tot1_1[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            self.unitary_U = unitary_Ut @ self.unitary_U

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot2_1[jj]), jump_ops, data_only=False, chi=None)).data.toarray()  # Liouvillian calc
            self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            unitary_Ut = la.expm(-1j * self.H_tot2_1[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            self.unitary_U = unitary_Ut @ self.unitary_U

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot1_2[jj]), jump_ops, data_only=False, chi=None)).data.toarray()  # Liouvillian calc
            self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            unitary_Ut = la.expm(-1j * self.H_tot1_2[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            self.unitary_U = unitary_Ut @ self.unitary_U

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot2_2[jj]), jump_ops, data_only=False, chi=None)).data.toarray()  # Liouvillian calc
            self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            unitary_Ut = la.expm(-1j * self.H_tot2_2[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            self.unitary_U = unitary_Ut @ self.unitary_U

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot1_3[jj]), jump_ops, data_only=False, chi=None)).data.toarray()  # Liouvillian calc
            self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            unitary_Ut = la.expm(-1j * self.H_tot1_3[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            self.unitary_U = unitary_Ut @ self.unitary_U

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot2_3[jj]), jump_ops, data_only=False, chi=None)).data.toarray()  # Liouvillian calc
            self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            unitary_Ut = la.expm(-1j * self.H_tot2_3[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            self.unitary_U = unitary_Ut @ self.unitary_U

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot1_4[jj]), jump_ops, data_only=False, chi=None)).data.toarray()  # Liouvillian calc
            self.L_array.append(L)
            Ut = la.expm(self.final_time / num_time_bins * L)  # time evolution (propagation operator)
            unitary_Ut = la.expm(-1j * self.H_tot1_4[jj] * self.final_time / num_time_bins)
            self.U = Ut @ self.U  # calculate total propagation until the time we are at
            self.unitary_U = unitary_Ut @ self.unitary_U

        # Reward and fidelity calculation
        fidelity = self.compute_fidelity()
        # if fidelity>0.49:
            # plot_complex_matrix(self.U, action, fidelity, "Matrix U")
            
        reward = (-3 * np.log10(1.0000001 - fidelity) + np.log10(1.0000001 - self.prev_fidelity)) + (3 * fidelity - self.prev_fidelity)
        # reward = (-7 * np.log10(1.0000001 - fidelity) + np.log10(1.0000001 - self.prev_fidelity)) + (3 * fidelity - self.prev_fidelity)
        # reward = (-1 * np.log10(1.0000001 - fidelity) + np.log10(1.0000001 - self.prev_fidelity)) + (3 * fidelity - self.prev_fidelity)
        self.prev_fidelity = fidelity

        self.state = self.get_observation()

        if self.current_Haar_num == self.num_Haar_basis:
            self.transition_history.append([fidelity, reward, *action, *self.U.flatten()])
       
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
    
    def canonicalDecomposition(self):
        
        ## This part of the code is from https://github.com/mpham26uchicago/laughing-umbrella/
        
        def decompose_one_qubit_product(
            U: np.ndarray, validate_input: bool = True, atol: float = 1e-8, rtol: float = 1e-5
        ):
            i, j = np.unravel_index(np.argmax(U, axis=None), U.shape)

            def u1_set(i):
                return (1, 3) if i % 2 else (0, 2)

            def u2_set(i):
                return (0, 1) if i < 2 else (2, 3)

            u1 = U[np.ix_(u1_set(i), u1_set(j))]
            u2 = U[np.ix_(u2_set(i), u2_set(j))]
            
            u1 = to_su(u1)
            u2 = to_su(u2)

            phase = U[i, j] / (u1[i // 2, j // 2] * u2[i % 2, j % 2])

            return phase, u1, u2
        
        def to_su(u: np.ndarray) -> np.ndarray:

            return u * complex(np.linalg.det(u)) ** (-1 / np.shape(u)[0])
        
        def KAK_2q(
            U: np.ndarray,
            rounding: int = 19
        ) -> Tuple[float, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, float,
                float, float]:

            # 0. Map U(4) to SU(4) (and phase)
            U = U / np.linalg.det(U)**0.25 

            assert np.isclose(np.linalg.det(U), 1), "Determinant of U is not 1"

            # 1. Unconjugate U into the magic basis
            B = 1 / np.sqrt(2) * np.array([[1., 0, 0, 1.j], [0, 1.j, 1., 0],
                                        [0, 1.j, -1., 0], [1., 0, 0, -1.j]]) # Magic Basis
            
            U_prime = np.conj(B).T @ U @ B

            # Isolating the maximal torus
            Theta = lambda U: np.conj(U)
            M_squared = Theta(np.conj(U_prime).T) @ U_prime
            
            if rounding is not None:
                M_squared = np.round(M_squared, rounding)  # For numerical stability

            ## 2. Diagonalizing M^2
            D, P = np.linalg.eig(M_squared)
            
            ## Check and correct for det(P) = -1
            if np.isclose(np.linalg.det(P), -1):
                P[:, 0] *= -1  # Multiply the first eigenvector by -1

            # 3. Extracting K2
            K2 = np.conj(P).T

            assert np.allclose(K2 @ K2.T, np.identity(4)), "K2 is not orthogonal"
            assert np.isclose(np.linalg.det(K2), 1), "Determinant of K2 is not 1"

            # 4. Extracting A
            A = np.sqrt(D)
            
            ## Check and correct for det(A) = -1
            if np.isclose(np.prod(A), -1):
                A[0] *= -1  # Multiply the first eigenvalue by -1

            A = np.diag(A)  # Turn the list of eigenvalues into a diagonal matrix
            
            assert np.isclose(np.linalg.det(A), 1), "Determinant of A is not 1"
            
            # 5. Extracting K1
            K1 = U_prime @ np.conj(K2).T @ np.conj(A).T
            
            assert np.allclose(K1 @ K1.T, np.identity(4)), "K1 is not orthogonal"
            assert np.isclose(np.linalg.det(K1), 1), "Determinant of K1 is not 1"

            # 6. Extracting Local Gates
            L = B @ K1 @ np.conj(B).T  # Left Local Product
            R = B @ K2 @ np.conj(B).T  # Right Local Product
            
            phase1, L1, L2 = decompose_one_qubit_product(L)  # L1 (top), L2(bottom)
            phase2, R1, R2 = decompose_one_qubit_product(R)  # R1 (top), R2(bottom)

            # 7. Extracting the Canonical Parameters
            C = np.array([[1, 1, 1], [-1, 1, -1], [1, -1, -1]])  # Coefficient Matrix

            theta_vec = np.angle(np.diag(A))[:3]  # theta vector
            a0, a1, a2 = np.linalg.inv(C) @ theta_vec  # Computing the "a"-vector

            # 8. Unpack Parameters and Put into Weyl chamber
            c0, c1, c2 = 2*a1, -2*a0, 2*a2 # Unpack parameters
            
            CAN = lambda c0, c1, c2: expm(1j/2*(c0*np.kron(X, X) + c1*np.kron(Y, Y) + c2*np.kron(Z, Z)))
            
            assert np.allclose(U, (phase1 * np.kron(L1, L2)) @ CAN(c0, c1, c2)
                            @ (phase2 * np.kron(R1, R2)), atol=1e-03), "U does not equal KAK"
            
            return phase1, L1, L2, phase2, R1, R2, c0, c1, c2
                        
        return KAK_2q(self.unitary_U_target)

    def KakActionCalculation(self):
        
        phase1, L1, L2, phase2, R1, R2, c0, c1, c2 = self.canonicalDecomposition()
        
        initialActions = np.zeros(27)
        
        initialActions[0] = self.singleQubitActionCalculation(R1)[0]
        initialActions[1] = self.singleQubitActionCalculation(R2)[0]
        initialActions[2] = self.singleQubitActionCalculation(R1)[1]
        initialActions[3] = self.singleQubitActionCalculation(R2)[1]
        initialActions[4] = self.singleQubitActionCalculation(R1)[2]
        initialActions[5] = self.singleQubitActionCalculation(R2)[2]
        
        initialActions[6] = self.canonicalActionCalculation(c0,c1,c2,1)
        
        initialActions[7]  = self.singleQubitActionCalculation(H)[0]
        initialActions[8]  = self.singleQubitActionCalculation(H)[0]
        initialActions[9]  = self.singleQubitActionCalculation(H)[1]
        initialActions[10]  = self.singleQubitActionCalculation(H)[1]
        initialActions[11]  = self.singleQubitActionCalculation(H)[2]
        initialActions[12]  = self.singleQubitActionCalculation(H)[2]
        
        initialActions[13] = self.canonicalActionCalculation(c0,c1,c2,2)
        
        initialActions[14]  = self.singleQubitActionCalculation(H@S@H)[0]
        initialActions[15]  = self.singleQubitActionCalculation(H@S@H)[0]
        initialActions[16]  = self.singleQubitActionCalculation(H@S@H)[1]
        initialActions[17]  = self.singleQubitActionCalculation(H@S@H)[1]
        initialActions[18]  = self.singleQubitActionCalculation(H@S@H)[2]
        initialActions[19]  = self.singleQubitActionCalculation(H@S@H)[2]
        
        initialActions[20] = self.canonicalActionCalculation(c0,c1,c2,3)
        
        initialActions[21] = self.singleQubitActionCalculation(L1@Sdagger@H)[0]
        initialActions[22] = self.singleQubitActionCalculation(L2@Sdagger@H)[0]
        initialActions[23] = self.singleQubitActionCalculation(L1@Sdagger@H)[1]
        initialActions[24] = self.singleQubitActionCalculation(L2@Sdagger@H)[1]
        initialActions[25] = self.singleQubitActionCalculation(L1@Sdagger@H)[2]
        initialActions[26] = self.singleQubitActionCalculation(L2@Sdagger@H)[2]
        
        return initialActions
    
    def singleQubitActionCalculation(self, U):
        
        singleQubitActions = np.zeros(3)
        
        x_angle , z_angle_after, z_angle_before = OneQubitEulerDecomposer(basis='ZXZ').angles(U)

        singleQubitActions[0] = np.mod((z_angle_after + z_angle_before) / np.pi, 2)  ## alpha
        singleQubitActions[1] = np.mod(x_angle / np.pi,2) ## gamma_magnitude
        singleQubitActions[2] = np.mod(- z_angle_before / np.pi,2)  ## gamma_phase
        
        return singleQubitActions
    
    def Rn(self, theta, axisSelection):
        return np.cos(theta/2)*I-1j*np.sin(theta/2)*axisSelection
    
    def ZXZ_Rotation_Generation(self, angles):
        return self.unitary_normalization(self.Rn(angles[1],Z)@self.Rn(angles[0],X)@self.Rn(angles[2],Z))
    
    def unitary_normalization(self, unitary_in):
        if unitary_in[0][0]==0:
            unitary_in = unitary_in/unitary_in[0][1]
        else:
            unitary_in = unitary_in/unitary_in[0][0]
            
        return unitary_in
    
    
    def canonicalActionCalculation(self, c0, c1, c2, index=1):
        
        twoQubitAction = 0
        
        if index == 1:
            b = 1/2*(c0+c1-c2)
        elif index ==2:
            b = 1/2*(-c0+c1+c2)
        elif index ==3:
            b = 1/2*(c0-c1+c2)
        else:
            print("wrong input index")
            
        twoQubitAction = - b / np.pi

        return twoQubitAction
