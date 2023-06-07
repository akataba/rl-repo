import os
import datetime

import gymnasium as gym
import numpy as np
import scipy.linalg as la
import cmath
from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj
from qutip.operators import *
# from relaqs.visualization.RealTimeScatterPlot import RealTimeScatterPlot

sig_p = np.array([[0,1],[0,0]])
sig_m = np.array([[0,0],[1,0]])
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])
Y = np.array([[0, 1j],[-1j, 0]])

class GateSynthEnvRLlibHaar(gym.Env):
    fidelities = []
    rewards = []
    # scatter_plot = RealTimeScatterPlot()

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
        self.final_time = env_config["final_time"] # Final time for the gates
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(env_config["observation_space_size"],))
        self.action_space = gym.spaces.Box(low=np.array([-0.1, 0, -np.pi]), high=np.array([0.1, 10, np.pi])) 
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
        self.prev_fidelity = 0
    
    def reset(self, *, seed=None, options=None):
        self.U = self.U_initial
        starting_observeration = self.unitary_to_observation(self.U_initial)
        self.current_Haar_num = 0
        self.H_array = []
        self.H_tot = []
        self.U_array = []
        self.prev_fidelity = 0
        info = {}
        return starting_observeration, info

    def step(self, action):
        truncated = False
        info = {}

        self.current_Haar_num += 1
        num_time_bins = 2 ** (self.current_Haar_num - 1)
        self.U = self.U_initial

        # Get actions
        alpha = action[0]
        gamma_magnitude = action[1]
        gamma_phase = action[2] 

        H = self.hamiltonian(self.delta, alpha, gamma_magnitude, gamma_phase)
        self.H_array.append(H)

        #array_temp = np.zeros((num_time_bins, 2, 2))
        self.H_tot = []

        ## Original
        # for ii in range(0, len(H_array)):
        #     for jj in range(0, 2 ** (self.current_Haar_num - 1)):
        #         if ii == 0:
        #             H_tot.append((-1) ** np.floor(jj / (2 ** (self.current_Haar_num - ii - 1))) * H_array[ii])
        #         else:
        #             H_tot[jj] += (-1) ** np.floor(jj / (2 ** (self.current_Haar_num - ii - 1))) * H_array[ii]

        ## Pythonic
        for ii, H_elem in enumerate(self.H_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - ii
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num-1)))
                if ii > 0:
                    self.H_tot[jj] += factor * H_elem 
                else:
                    self.H_tot.append(factor * H_elem)


        for jj in range(0, num_time_bins):
            Ut = la.expm(-1j* self.final_time/num_time_bins *self.H_tot[jj])
            self.U = Ut @ self.U 

        self.state = self.unitary_to_observation(self.U)
        self.U_array.append(self.U)

        # Get reward (fidelity)
        fidelity = float(np.abs(np.trace(self.U_target.conjugate().transpose()@self.U)))  / (self.U.shape[0])
        reward = -(np.log10(1.0-fidelity)-np.log10(1.0-self.prev_fidelity))
        self.prev_fidelity = fidelity

        GateSynthEnvRLlibHaar.append_fidelity(fidelity)
        GateSynthEnvRLlibHaar.append_reward(reward)

        if len(GateSynthEnvRLlibHaar.get_fidelities()) % 300 == 0:
            GateSynthEnvRLlibHaar.save_data()

        # if self.current_Haar_num == self.num_Haar_basis:
        #     GateSynthEnvRLlibHaarNoiseless.scatter_plot.plot(GateSynthEnvRLlibHaarNoiseless.get_fidelities(), GateSynthEnvRLlibHaarNoiseless.get_rewards())

        # Determine if episode is over
        truncated = False
        terminated = False
        if self.current_Haar_num >= self.num_Haar_basis:
            truncated = True
        elif (fidelity >= 1):
            terminated = True
        else:
            terminated = False

        return (self.state, reward, terminated, truncated, info)

    def unitary_to_observation(self, U):
       return np.array([(abs(x), cmath.phase(x)/np.pi) for x in U.flatten()], dtype=np.float64).squeeze().reshape(-1)
    
    def hamiltonian(self, delta, alpha, gamma_magnitude, gamma_phase):
        """Alpha and gamma are complex. This function could be made a callable class attribute."""
        return (delta + alpha)*Z + gamma_magnitude*(np.cos(gamma_phase)*X + np.sin(gamma_phase)*Y)

    @classmethod    
    def append_fidelity(cls,fidelity):
        cls.fidelities.append(fidelity)

    @classmethod
    def append_reward(cls,reward):
        cls.rewards.append(reward)

    @classmethod
    def get_fidelities(cls):
        return cls.fidelities

    @classmethod    
    def get_rewards(cls):
        return cls.rewards
    
    @classmethod
    def save_data(cls):
        # Get the next file number
        file_num = cls.get_next_file_number()

        # Get the data to be saved
        fidelity_data = cls.get_fidelities()
        reward_data = cls.get_rewards()

        # Create a file name
        file_name = f"data-{file_num:03}.txt"

        # Set the file path
        file_dir = "../results/"
        file_path = os.path.join(file_dir, file_name)

        # Save the data to the file
        with open(file_path, "w") as file:
            for fidelity, reward in zip(fidelity_data, reward_data):
                file.write(f"{fidelity},{reward}\n")

        print(f"Data saved to: {file_path}")

    @classmethod
    def get_next_file_number(cls):
        file_dir = "../results/"

        # Get the existing file numbers
        existing_files = []
        for file_name in os.listdir(file_dir):
            if file_name.startswith("data-") and file_name.endswith(".txt"):
                file_num = int(file_name[5:-4])
                existing_files.append(file_num)

        # Find the next file number
        if existing_files:
            next_file_num = max(existing_files) + 1
        else:
            next_file_num = 1

        return next_file_num
    


class GateSynthEnvRLlibHaarNoisy(gym.Env):
    fidelities = []
    rewards = []

    data_dir = "../results/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")

    @classmethod
    def get_default_env_config(cls):
        return {
            "observation_space_size": 32,
            "action_space_size": 3,
            "U_initial": (spre(Qobj(I))*spost(Qobj(I))).data.toarray(),
            "U_target" : (spre(Qobj(X))*spost(Qobj(X))).data.toarray(),
            "dt" : 0.001,
            "final_time": 0.3,
            "num_Haar_basis": 1,
            "delta": 0,
        }
 
    def __init__(self, env_config):
        self.dt = env_config["dt"]
        self.final_time = env_config["final_time"] # Final time for the gates
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(env_config["observation_space_size"],))
        self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1])) 
        self.delta = env_config["delta"] # detuning
        self.U_target = env_config["U_target"]
        self.U_initial = env_config["U_initial"] 
        self.num_Haar_basis = env_config["num_Haar_basis"]
        self.current_Haar_num = 0
        self.H_array = []
        self.H_tot = []
        self.L_array = []
        self.U_array = []
        self.U = []
        self.state = self.unitary_to_observation(self.U_initial)
        self.prev_fidelity = 0
    
    def reset(self, *, seed=None, options=None):
        starting_observeration = self.unitary_to_observation(self.U_initial)
        self.current_Haar_num = 0
        self.H_array = []
        self.H_tot = []
        self.L_array = []
        self.U_array = []
        self.U = []
        # self.state = self.unitary_to_observation(self.U_initial) ## DO WE NEED THIS?
        self.prev_fidelity = 0
        info = {}
        return starting_observeration, info

    def step(self, action):
        truncated = False
        info = {}

        self.current_Haar_num += 1
        num_time_bins = 2 ** (self.current_Haar_num - 1)
        self.U = self.U_initial

        # Get actions
        alpha = 0.1*action[0]
        gamma_magnitude = 3*(action[1]+1)
        gamma_phase = 1.1*np.pi*action[2] 

        # Set noise opertors
        relaxationRate = 0.01
        jump_ops = [np.sqrt(relaxationRate)*sigmam()]

        H = self.hamiltonian(self.delta, alpha, gamma_magnitude, gamma_phase)
        self.H_array.append(H)

        self.H_tot = []

        for ii, H_elem in enumerate(self.H_array):
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - ii
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num-1)))
                if ii > 0:
                    self.H_tot[jj] += factor * H_elem 
                else:
                    self.H_tot.append(factor * H_elem)

        t = 0
        self.U = np.eye(4)

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot[jj]), jump_ops, data_only=False, chi=None)).data.toarray()
            self.L_array.append(L)
            Ut = la.expm(self.final_time/num_time_bins * L)
            self.U = Ut @ self.U 

        self.state = self.unitary_to_observation(self.U)

        # Get reward (fidelity)
        fidelity = float(np.abs(np.trace(self.U_target.conjugate().transpose()@self.U)))  / (self.U.shape[0])
        reward = (-np.log10(1.0-fidelity)+np.log10(1.0-self.prev_fidelity))+(fidelity-self.prev_fidelity)
        self.prev_fidelity = fidelity

        print(self.current_Haar_num,fidelity,alpha,gamma_magnitude,gamma_phase)

        GateSynthEnvRLlibHaarNoisy.append_fidelity(fidelity)
        GateSynthEnvRLlibHaarNoisy.append_reward(reward)

        if len(GateSynthEnvRLlibHaarNoisy.get_fidelities()) % 300 == 0:
            GateSynthEnvRLlibHaarNoisy.save_data()

        # if self.current_Haar_num == self.num_Haar_basis:
        #     GateSynthEnvRLlibHaarNoiseless.scatter_plot.plot(GateSynthEnvRLlibHaarNoiseless.get_fidelities(), GateSynthEnvRLlibHaarNoiseless.get_rewards())

        # Determine if episode is over
        truncated = False
        terminated = False
        if self.current_Haar_num >= self.num_Haar_basis:
            truncated = True
        elif (fidelity >= 1):
            terminated = True
        else:
            terminated = False

        return (self.state, reward, terminated, truncated, info)

    def unitary_to_observation(self, U):
       return np.array([(abs(x), (cmath.phase(x)/np.pi+1)/2) for x in U.flatten()], dtype=np.float64).squeeze().reshape(-1)
    
    def hamiltonian(self, delta, alpha, gamma_magnitude, gamma_phase):
        """Alpha and gamma are complex. This function could be made a callable class attribute."""
        return (delta + alpha)*Z + gamma_magnitude*(np.cos(gamma_phase)*X + np.sin(gamma_phase)*Y)

    @classmethod    
    def append_fidelity(cls,fidelity):
        cls.fidelities.append(fidelity)

    @classmethod
    def append_reward(cls,reward):
        cls.rewards.append(reward)

    @classmethod
    def get_fidelities(cls):
        return cls.fidelities

    @classmethod    
    def get_rewards(cls):
        return cls.rewards
    
    @classmethod
    def save_data(cls):
        # Get the next file number
        file_num = cls.get_next_file_number()

        # Get the data to be saved
        fidelity_data = cls.get_fidelities()
        reward_data = cls.get_rewards()

        # Create a file name
        file_name = f"data-{file_num:03}.txt"

        # Set the file path
        file_path = os.path.join(cls.data_dir, file_name)

        # Save the data to the file
        with open(file_path, "w") as file:
            for fidelity, reward in zip(fidelity_data, reward_data):
                file.write(f"{fidelity},{reward}\n")

        print(f"Data saved to: {file_path}")

    @classmethod
    def get_next_file_number(cls):
        # Get the existing file numbers
        existing_files = []

        if not os.path.exists(cls.data_dir):
                    os.mkdir(cls.data_dir)

        for file_name in os.listdir(cls.data_dir):
            if file_name.startswith("data-") and file_name.endswith(".txt"):
                file_num = int(file_name[5:-4])
                existing_files.append(file_num)

        # Find the next file number
        if existing_files:
            next_file_num = max(existing_files) + 1
        else:
            next_file_num = 1

        return next_file_num