""" """
from asyncore import file_dispatcher
import gymnasium as gym
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


sig_p = np.array([[0,1],[0,0]])
sig_m = np.array([[0,0],[1,0]])
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])
Y = np.array([[0, 1j],[-1j, 0]])

class GateSynthEnvRLlib(gym.Env):
    @classmethod
    def get_default_env_config(cls):
        return {
            "observation_space_size": 8,
            "action_space_size": 3,
            "U_initial": I,
            "U_target" : X,
            "final_time": 2,
            "dt": 0.01,
            "delta": 1,
        }
 
    def __init__(self, env_config):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(env_config["observation_space_size"],))
        self.action_space = gym.spaces.Box(low=-1/np.sqrt(2), high=1/np.sqrt(2), shape=(env_config["action_space_size"],)) # assuming norm should be <= 1
        self.t = 0
        self.final_time = env_config["final_time"] # Final time for the gates
        self.dt = env_config["dt"]  # time step
        self.delta = env_config["delta"] # detuning
        self.U_target = env_config["U_target"]
        self.U_initial = env_config["U_initial"] # future todo, can make random initial state
        self.U = env_config["U_initial"]
        self.state = self.unitary_to_observation(self.U)
        self.amplitudes =[]

    
    def reset(self, *, seed=None, options=None):
        self.t = 0
        self.U = self.U_initial
        starting_observeration = self.unitary_to_observation(self.U_initial)
        info = {}
        self.amplitudes = []
        return starting_observeration, info

    def step(self, action):
        truncated = False
        info = {}

        # Get actions
        alpha = action[0]
        gamma = action[1] + 1j * action[2]
        
        # Get state
        H = self.hamiltonian(self.delta, alpha, gamma)
        Ut = la.expm(-1j*self.dt*H)
        self.U = Ut @ self.U # What is the purpose of this operation ?
        self.state = self.unitary_to_observation(self.U)

        #leaving off conjugate transpose since X yields itself : <--- which line did this refer to?
        # Get reward (fidelity)
        fidelity = 0.5 *  abs(np.trace(self.U_target@self.U))**2
        reward = fidelity 
        
        # Determine if episode is over
        truncated = False
        if (fidelity >= 0.95) or self.t >= self.final_time:
            terminated = True
            truncated = True
        elif self.t >= self.final_time:
            terminated = True
        else:
            terminated = False

        self.t = self.t + self.dt # increment time
        self.amplitudes.append([alpha, gamma, fidelity])

        return (self.state, reward, terminated, truncated, info)

    def unitary_to_observation(self, U):
       return np.clip(np.array([(x.real, x.imag) for x in U.flatten()], dtype=np.float64).squeeze().reshape(-1), -1, 1)
    
    def hamiltonian(self, delta, alpha, gamma):
        """Alpha and gamma are complex. This function could be made a callable class attribute."""
        return alpha*Z + 0.5*(gamma*sig_m + gamma.conjugate()*sig_p) + delta*Z
    
    def get_fidelity(self):
        fidelity = [self.amplitudes[i][2] for i in range(self.amplitudes)]
        return fidelity
  




