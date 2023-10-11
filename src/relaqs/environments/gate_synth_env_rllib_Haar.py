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

class GateSynthEnvRLlibHaar(gym.Env):
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
            "save_data_every_step": 1,
            "verbose": True,
            "observation_space_size": 9,  # 1 (fidelity) + 8 (flattened unitary)
        }
    def __init__(self, env_config):
        self.final_time = env_config["final_time"]  # Final time for the gates
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(env_config["observation_space_size"],))
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))
        self.delta = env_config["delta"]  # detuning
        self.U_target = env_config["U_target"]
        self.U_initial = env_config["U_initial"] # future todo, can make random initial state
        self.U = env_config["U_initial"]
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
        self.gamma_detuning_max = 0.05E9      #detuning of the control pulse in Hz 
        self.transition_history = []

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
        return float(np.abs(np.trace(self.U_target.conjugate().transpose() @ self.U))) / (self.U.shape[0])

    def hamiltonian(self, delta, alpha, gamma_magnitude, gamma_phase):
        """Alpha and gamma are complex. This function could be made a callable class attribute."""
        return (delta + alpha) * Z + gamma_magnitude * (np.cos(gamma_phase) * X + np.sin(gamma_phase) * Y)

    def reset(self, *, seed=None, options=None):
        self.U = self.U_initial
        starting_observeration = self.get_observation()
        self.state = self.get_observation()
        self.current_Haar_num = 1
        self.current_step_per_Haar = 1
        self.H_array = []
        self.H_tot = []
        self.U_array = []
        self.prev_fidelity = 0
        info = {}
        return starting_observeration, info

    def step(self, action):
        num_time_bins = 2 ** (self.current_Haar_num - 1) # Haar number decides the number of time bins

        # gamma is the complex amplitude of the control field
        gamma_magnitude = self.gamma_magnitude_max / 2 * (action[0] + 1)
        gamma_phase = self.gamma_phase_max * action[1]
        alpha = self.gamma_detuning_max * action[2]

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

        self.U = self.U_initial

        for jj in range(0, num_time_bins):
            Ut = la.expm(-1j * self.final_time / num_time_bins * self.H_tot[jj])
            self.U = Ut @ self.U

        self.U_array.append(self.U)

        # Get reward (fidelity)
        fidelity = self.compute_fidelity()
        reward = (-3 * np.log10(1.0 - fidelity) + np.log10(1.0 - self.prev_fidelity)) + (3 * fidelity - self.prev_fidelity)
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


class GateSynthEnvRLlibHaarNoisy(gym.Env):
    @classmethod
    def get_default_env_config(cls):
        return {
            # "action_space_size": 3,
            "action_space_size": 2,
            "U_initial": I,  # staring with I
            "U_target": X,  # target for X
            "final_time": 35.5556E-9, # in seconds
            "num_Haar_basis": 1,  # number of Haar basis (need to update for odd combinations)
            "steps_per_Haar": 2,  # steps per Haar basis per episode
            "delta": [0],  # qubit detuning
            "save_data_every_step": 1,
            "verbose": True,
#            "relaxation_rates_list": [[0.01,0.02],[0.05, 0.07]], # relaxation lists of list of floats to be sampled from when resetting environment.
#            "relaxation_ops": [sigmam(),sigmaz()] #relaxation operator lists for T1 and T2, respectively
            "relaxation_rates_list": [[314159]], # relaxation lists of list of floats to be sampled from when resetting environment. (10 usec)
            "relaxation_ops": [sigmam()], #relaxation operator lists for T1 and T2, respectively
#            "observation_space_size": 35, # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate
            "observation_space_size": 2*16 + 1 + 1 + 1 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 1 for relaxation rate + 1 for detuning
        }

    def __init__(self, env_config):
        self.final_time = env_config["final_time"]  # Final time for the gates
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(env_config["observation_space_size"],))  # propagation operator elements + fidelity
        # self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1])) # for detuning included control
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]))
#        self.delta = [env_config["delta"]]  # detuning
        self.delta = env_config["delta"]  # detuning
        self.detuning = 0
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
        self.gamma_phase_max = 1.1675 * np.pi
        self.gamma_magnitude_max = 1.8 * np.pi / self.final_time / self.steps_per_Haar
        self.transition_history = []

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
    
    def compute_fidelity(self):
        env_config = GateSynthEnvRLlibHaarNoisy.get_default_env_config()
        U_target_dagger = self.unitary_to_superoperator(env_config["U_target"].conjugate().transpose())
        return float(np.abs(np.trace(U_target_dagger @ self.U))) / (self.U.shape[0])

    def unitary_to_observation(self, U):
        return (
            np.array(
                [(abs(x), (cmath.phase(x) / np.pi + 1) / 2) for x in U.flatten()], 
                dtype=np.float64,
                )
            .squeeze()
            .reshape(-1)  # cmath phase gives -pi to pi
        )

    def hamiltonian(self, delta, alpha, gamma_magnitude, gamma_phase):
        """Alpha and gamma are complex. This function could be made a callable class attribute."""
        return (delta + alpha) * Z + gamma_magnitude * (np.cos(gamma_phase) * X + np.sin(gamma_phase) * Y)

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
        return starting_observeration, info

    def step(self, action):
        num_time_bins = 2 ** (self.current_Haar_num - 1) # Haar number decides the number of time bins

        # action space setting
        alpha = 0  # in current simulation we do not adjust the detuning

        # gamma is the complex amplitude of the control field
        gamma_magnitude = self.gamma_magnitude_max / 2 * (action[0] + 1)
        gamma_phase = self.gamma_phase_max * action[1]

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
                "Step: ", f"{self.current_step_per_Haar}",
                "Relaxation rates:")
            for rate in self.relaxation_rate:
                print(f"{rate:7.6f}")
            print(
                "F: ", f"{fidelity:7.3f}",
                "R: ", f"{reward:7.3f}",
                "amp: " f"{action[0]:7.3f}",
                "phase: " f"{action[1]:7.3f}",
            )

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


class TwoQubitGateSynth(gym.Env):
    @classmethod

    #physics: https://journals.aps.org/prapplied/pdf/10.1103/PhysRevApplied.10.054062, eq(2)
    #parameters: https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.021058
    #30 ns duration, g1 = 72.5 MHz, g2 = 71.5 MHz, g12 = 5 MHz
    #T1 = 60 us, 30 us
    #T2* = 66 us, 5 us

    def hamiltonian(self, delta1, delta2, alpha1, alpha2, alphaC, gamma_magnitude1, gamma_phase1, gamma_magnitude2, gamma_phase2, g1 = 72.5E6, g2 = 71.5E6, g12 = 5E6):
        selfEnergyTerms = (delta1 + alpha1) * Z1 + (delta2 + alpha2) * Z2
        Qubit1ControlTerms = gamma_magnitude1 * (np.cos(gamma_phase1) * X1 + np.sin(gamma_phase1) * Y1)
        Qubit2ControlTerms = gamma_magnitude2 * (np.cos(gamma_phase2) * X1 + np.sin(gamma_phase2) * Y1)

        #omega1 = delta1+alpha1, omega2 = delta2+alpha2, omegaC = alphaC
        Delta1 = delta1+alpha1-alphaC
        Delta2 = delta2+alpha2-alphaC
        twoQubitDetuning = (1/Delta1 + 1/Delta2)/2
        
        g_eff = g1*g2/twoQubitDetuning + g12
        interactionEnergy = g_eff*exchangeOperator

        energyTotal = selfEnergyTerms + interactionEnergy

        return energyTotal

    def get_default_env_config(cls):
        return {
            "action_space_size": 2,
            "U_initial": I,  # staring with I
            "U_target": CZ,  # target for X
            "final_time": 30E-9, # in seconds
            "num_Haar_basis": 3,  # number of Haar basis (need to update for odd combinations)
            "steps_per_Haar": 2,  # steps per Haar basis per episode
            "delta": [[0],[0]],  # qubit detuning
            "save_data_every_step": 1,
            "verbose": True,
            "relaxation_rates_list": [[1/60E-6/2/np.pi],[1/30E-6/2/np.pi],[1/66E-6/2/np.pi],[1/5E-6/2/np.pi]], # relaxation lists of list of floats to be sampled from when resetting environment.
            "relaxation_ops": [sigmam1,sigmam2,Qobj(Z1),Qobj(Z2)], #relaxation operator lists for T1 and T2, respectively
#            "observation_space_size": 35, # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate
            "observation_space_size": 2*256 + 1 + 4 + 2 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 4 for relaxation rate + 2 for detuning
        }

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
        self.alpha_max = 0.05E9
        self.alphaC_mod_max = 1.5E9  ## see https://journals.aps.org/prx/pdf/10.1103/PhysRevX.11.021058
        self.alphaC0 = 1.04E9 # couper center frequency : 5.2GHz, qubit 1 center frequency: 4.16 GHz
        self.gamma_phase_max = 1.1675 * np.pi
        self.gamma_magnitude_max = 1.8 * np.pi / self.final_time / self.steps_per_Haar
        self.transition_history = []

    def detuning_update(self):
        # Random detuning selection
        if len(self.delta[0])==1:
            detuning1 = self.delta[0]
        else:
            detuning1 = random.sample(self.delta[0],k=1)[0]
            
        # Random detuning selection
        if len(self.delta[1])==1:
            detuning2 = self.delta[0]
        else:
            detuning2 = random.sample(self.delta[1],k=1)[0]

        self.detuning = [detuning1, detuning2]
        
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
        normalizedDetuning = [(self.detuning[0] - min(self.delta[0])+1E-15)/(max(self.delta[0])-min(self.delta[0])+1E-15), (self.detuning[1] - min(self.delta[1])+1E-15)/(max(self.delta[1])-min(self.delta[1])+1E-15)]
        return np.append([self.compute_fidelity()]+[x//6283185 for x in self.relaxation_rate]+normalizedDetuning, self.unitary_to_observation(self.U)) #6283185 assuming 500 nanosecond relaxation is max
    
    def compute_fidelity(self):
        env_config = TwoQubitGateSynth.get_default_env_config()
        U_target_dagger = self.unitary_to_superoperator(env_config["U_target"].conjugate().transpose())
        return float(np.abs(np.trace(U_target_dagger @ self.U))) / (self.U.shape[0])

    def unitary_to_observation(self, U):
        return (
            np.array(
                [(abs(x), (cmath.phase(x) / np.pi + 1) / 2) for x in U.flatten()], 
                dtype=np.float64,
                )
            .squeeze()
            .reshape(-1)  # cmath phase gives -pi to pi
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
        return starting_observeration, info

    def step(self, action):
        num_time_bins = 2 ** (self.current_Haar_num - 1) # Haar number decides the number of time bins

        ### action space setting
        alpha1 = self.alpha_max * action[0] 
        alpha2 = self.alpha_max * action[1] 
        alphaC = self.alphaC0 + self.alphaC_mod_max * action[2] 

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
        H = self.hamiltonian(self.delta[0][0], self.delta[1][0], alpha1, alpha2, alphaC, gamma_magnitude1, gamma_phase1, gamma_magnitude2, gamma_phase2):
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
        reward = (-3 * np.log10(1.0 - fidelity) + np.log10(1.0 - self.prev_fidelity)) + (3 * fidelity - self.prev_fidelity)
        self.prev_fidelity = fidelity

        self.state = self.get_observation()

        # printing on the command line for quick viewing
        # if self.verbose is True:
        #     print(
        #         "Step: ", f"{self.current_step_per_Haar}",
        #         "Relaxation rates:")
        #     for rate in self.relaxation_rate:
        #         print(f"{rate:7.6f}")
        #     print(
        #         "F: ", f"{fidelity:7.3f}",
        #         "R: ", f"{reward:7.3f}",
        #         "amp: " f"{action[0]:7.3f}",
        #         "phase: " f"{action[1]:7.3f}",
        #     )

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


