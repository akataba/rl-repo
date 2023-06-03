import gymnasium as gym
import numpy as np
import scipy.linalg as la
from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj
from qutip.operators import *


sig_p = np.array([[0,1],[0,0]])
sig_m = np.array([[0,0],[1,0]])
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])
Y = np.array([[0, 1j],[-1j, 0]])

class GateSynthEnvRLlibHaarNoiseless(gym.Env):
    @classmethod
    def get_default_env_config(cls):
        return {
            "observation_space_size": 8,
            "action_space_size": 3,
            "U_initial": I,
            "U_target" : X,
            "final_time": 0.3,
            "num_Haar_basis": 5,
            "delta": 0,
        }
 
    def __init__(self, env_config):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(env_config["observation_space_size"],))
        self.action_space = gym.spaces.Box(low=np.array([-0.1, 0.1, -1.1*np.pi]), high=np.array([0.1, 10, 1.1*np.pi])) 
        self.t = 0
        self.final_time = env_config["final_time"] # Final time for the gates
        self.delta = env_config["delta"] # detuning
        self.U_target = env_config["U_target"]
        self.U_initial = env_config["U_initial"] # future todo, can make random initial state
        self.U = env_config["U_initial"]
        self.num_Haar_basis = env_config["num_Haar_basis"]
        self.current_Haar_num = 0
        self.H_array = []
        self.H_tot = []
        self.U_array = []
        self.state = self.unitary_to_observation(self.U)
    
    def reset(self, *, seed=None, options=None):
        self.t = 0
        self.U = self.U_initial
        starting_observeration = self.unitary_to_observation(self.U_initial)
        info = {}
        return starting_observeration, info

    def step(self, action):
        truncated = False
        info = {}

        self.current_Haar_num += 1

        num_time_bins = 2 ** (self.current_Haar_num - 1)

        H = self.hamiltonian(self.delta, alpha, gamma_magnitude, gamma_phase)
        self.H_array.append(H)

        array_temp = np.zeros(num_time_bins)
        self.H_tot = Qobj(array_temp)

        ## Original
        # for ii in range(0, len(H_array)):
        #     for jj in range(0, 2 ** (self.current_Haar_num - 1)):
        #         if ii == 0:
        #             H_tot.append((-1) ** np.floor(jj / (2 ** (self.current_Haar_num - ii - 1))) * H_array[ii])
        #         else:
        #             H_tot[jj] += (-1) ** np.floor(jj / (2 ** (self.current_Haar_num - ii - 1))) * H_array[ii]

        ## Pythonic
        for ii, H_elem in enumerate(H_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - ii
                factor = (-1) ** np.floor(jj / (2 ** Haar_num))
                self.H_tot[jj] += factor * H_elem


        for jj in range(0, num_time_bins):
            Ut = la.expm(-1j* self.final_time/num_time_bins *self.H_tot[jj])
            self.U = Ut @ self.U 

        self.state = self.unitary_to_observation(self.U)
        self.U_array.append(self.U)

        # Get reward (fidelity)
        fidelity = float(np.abs(np.trace(self.U_target.conjugate().transpose()@self.U)))  / (self.U.shape[0]**2)
        reward = -np.log10(1-fidelity)

        # Determine if episode is over
        truncated = False
        if (fidelity >= 0.95):
            terminated = True
            truncated = True
        elif self.current_Haar_num >= self.num_Haar_basis:
            terminated = True
        else:
            terminated = False

        return (self.state, reward, terminated, truncated, info)

    def unitary_to_observation(self, U):
       return np.clip(np.array([(x.real, x.imag) for x in U.flatten()], dtype=np.float64).squeeze().reshape(-1), -1, 1) # todo, see if clip is necessary
    
    def hamiltonian(self, delta, alpha, gamma_magnitude, gamma_phase):
        """Alpha and gamma are complex. This function could be made a callable class attribute."""
        return (delta + alpha)*Z + gamma_magnitude*(np.cos(gamma_phase)*X + np.sin(gamma_phase)*Y)
    


class GateSynthEnvRLlibHaarNoisy(gym.Env):
    @classmethod
    def get_default_env_config(cls):
        return {
            "observation_space_size": 32,
            "action_space_size": 3,
            "L_initial": (spre(Qobj(I))*spost(Qobj(I))).data.toarray(),
            "L_target" : (spre(Qobj(X))*spost(Qobj(X))).data.toarray(),
            "final_time": 0.3,
            "dt": 0.001,
            "delta": 0,
        }
 
    def __init__(self, env_config):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(env_config["observation_space_size"],))
        self.action_space = gym.spaces.Box(low=np.array([-0.1, 0.1, -1.1*np.pi]), high=np.array([0.1, 10, 1.1*np.pi])) # TODO: verify these bounds
        #self.action_space = gym.spaces.Box(low=[0, 0, 0], high=[1, 1/np.sqrt(2), 1/np.sqrt(2)], shape=(env_config["action_space_size"],))
        self.t = 0
        self.final_time = env_config["final_time"] # Final time for the gates
        self.dt = env_config["dt"]  # time step
        self.delta = env_config["delta"] # detuning
        self.L_target = env_config["L_target"]
        self.L_initial = env_config["L_initial"] 
        self.L = env_config["L_initial"]
        self.state = self.unitary_to_observation(self.L)
    
    def reset(self, *, seed=None, options=None):
        self.t = 0
        self.L = self.L_initial
        # starting_observeration = self.unitary_to_observation(self.U_initial)
        starting_observeration = self.unitary_to_observation(self.L_initial)
        info = {}
        return starting_observeration, info

    def step(self, action):
        truncated = False
        info = {}

        # Here, Rewards for Liouvillian should be used.
        fidelity = float(np.abs(np.trace(self.L @ self.L_target.conjugate().transpose())))  / self.L.shape[0]
        reward = -np.log10(1-fidelity)

        return (self.state, reward, terminated, truncated, info)

    def unitary_to_observation(self, U):
       return np.clip(np.array([(x.real, x.imag) for x in U.flatten()], dtype=np.float64).squeeze().reshape(-1), -1, 1) # todo, see if clip is necessary
    
    def liouvillianWithControl(self, delta, alpha, gamma_magnitude, gamma_phase, jump_ops):
        """This is Liouvillian so should be separately used from the Hamiltonian"""
        X = sigmax()
        Y = sigmay()
        Z = sigmaz()

        H = (delta + alpha)*Z + gamma_magnitude*(np.cos(gamma_phase)*X + np.sin(gamma_phase)*Y)

        L = liouvillian(H, jump_ops, data_only=False, chi=None)

        return L.data.toarray()