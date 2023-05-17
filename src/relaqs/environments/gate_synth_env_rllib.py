import gymnasium as gym
import numpy as np
import scipy.linalg as la

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
        self.action_space = gym.spaces.Box(low=np.array([0, 0, 0]), high=np.array([2, 10, np.pi])) # TODO: verify these bounds
        #self.action_space = gym.spaces.Box(low=[0, 0, 0], high=[1, 1/np.sqrt(2), 1/np.sqrt(2)], shape=(env_config["action_space_size"],))
        self.t = 0
        self.final_time = env_config["final_time"] # Final time for the gates
        self.dt = env_config["dt"]  # time step
        self.delta = env_config["delta"] # detuning
        self.U_target = env_config["U_target"]
        self.U_initial = env_config["U_initial"] # future todo, can make random initial state
        self.U = env_config["U_initial"]
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

        # Get actions
        alpha = action[0] 
        gamma_magnitude = action[1]
        gamma_phase = action[2] 
        
        # Get state
        H = self.hamiltonian(self.delta, alpha, gamma_magnitude, gamma_phase)
        Ut = la.expm(-1j*self.dt*H)
        self.U = Ut @ self.U # What is the purpose of this operation ?
        self.state = self.unitary_to_observation(self.U)

        #leaving off conjugate transpose since X yields itself : <--- which line did this refer to?
        # Get reward (fidelity)
        fidelity = float(np.abs(np.trace(self.U_target.conjugate().transpose()@self.U)))  / self.U.shape[0]
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

        return (self.state, reward, terminated, truncated, info)

    def unitary_to_observation(self, U):
       return np.clip(np.array([(x.real, x.imag) for x in U.flatten()], dtype=np.float64).squeeze().reshape(-1), -1, 1) # todo, see if clip is necessary
    
    def hamiltonian(self, delta, alpha, gamma_magnitude, gamma_phase):
        """Alpha and gamma are complex. This function could be made a callable class attribute."""
        #return alpha*Z + 0.5*(gamma*sig_m + gamma.conjugate()*sig_p) + delta*Z
        return (delta + alpha)*Z + gamma_magnitude*(np.cos(gamma_phase)*X + np.sin(gamma_phase)*Y)
    