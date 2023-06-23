import os
import datetime

import gymnasium as gym
import numpy as np
import scipy.linalg as la
import cmath
from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj
from qutip.operators import *

sig_p = np.array([[0,1],[0,0]])
sig_m = np.array([[0,0],[1,0]])
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])
Y = np.array([[0, 1j],[-1j, 0]])

class GateSynthEnvRLlibHaar(gym.Env):
    fidelities = []
    rewards = []

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
        self.transition_history = []
    
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

        self.H_tot = []

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

        self.transition_history.append([fidelity, reward, *action])

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

    gamma_magnitudes = []
    gamma_phases = []

    #data saving diretory
    data_dir = "../results/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")

    @classmethod
    def get_default_env_config(cls):
        return {
            "observation_space_size": 33,   #2*16 =  (complex number)*(density matrix elements = 4)^2, + 1 for fidelity
            # "action_space_size": 3,
            "action_space_size": 2,
            "U_initial": (spre(Qobj(I))*spost(Qobj(I))).data.toarray(),   #staring with I 
            "U_target" : (spre(Qobj(X))*spost(Qobj(X))).data.toarray(),   #target for X
            "final_time": 0.3,
            "num_Haar_basis": 1,                                          #number of Haar basis (need to update for odd combinations)
            "steps_per_Haar": 3,                                          #steps per Haar basis per episode 
            "delta": 0,                                                   #qubit detuning
            "save_data_every_step" : 1
        }
 
    def __init__(self, env_config):
        self.final_time = env_config["final_time"]                                                              #Final time for the gates
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(env_config["observation_space_size"],))   #propagation operator elements + fidelity
        # self.action_space = gym.spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]))              #for detuning included control
        self.action_space = gym.spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1])) 
        self.delta = env_config["delta"]                                                                        #detuning
        self.U_target = env_config["U_target"]
        self.U_initial = env_config["U_initial"] 
        self.num_Haar_basis = env_config["num_Haar_basis"]
        self.steps_per_Haar = env_config["steps_per_Haar"]
        self.current_Haar_num = 1                                                                               #starting with 1
        self.current_step_per_Haar = 1
        self.H_array = []                                                                                       #saving all H's with Haar wavelet to be multiplied
        self.H_tot = []                                                                                         #Haar wavelet multipied H summed up for each time bin
        self.L_array = []                                                                                       #Liouvillian for each time bin
        self.U_array = []                                                                                       #propagation operators for each time bin
        self.U = []                                                                                             #multiplied propagtion operators
        self.state = self.unitary_to_observation(self.U_initial)                                                #starting observation space
        self.prev_fidelity = 0                                                                                  #previous step' fidelity for rewarding
        self.gamma_phase_max = 1.1675*np.pi                                                                               
        self.gamma_magnitude_max = 1.8*np.pi/self.final_time/self.steps_per_Haar
        self.save_data_every_step = env_config["save_data_every_step"]
        self.transition_history = []
    
    def reset(self, *, seed=None, options=None):
        starting_observeration = self.unitary_to_observation(self.U_initial)
        self.state = self.unitary_to_observation(self.U_initial) 
        self.current_Haar_num = 1
        self.current_step_per_Haar = 1
        self.H_array = []
        self.H_tot = []
        self.L_array = []
        self.U_array = []
        self.U = []
        self.prev_fidelity = 0
        info = {}                                                               #Do we need this?
        return starting_observeration, info

    def step(self, action):

        num_time_bins = 2 ** (self.current_Haar_num - 1)                        #Haar number decides the number of time bins
        self.U = self.U_initial                                                 #At every step, we start with I, then calculate the propagator for all hamiltonians


        # action space setting
        alpha = 0               # in current simulation we do not adjust the detuning

        # gamma is the complex amplitude of the control field
        gamma_magnitude = self.gamma_magnitude_max/2*(action[0]+1)
        gamma_phase = self.gamma_phase_max*action[1]


        # action space setting with arctanh
        # alpha = 0.1*np.arctanh(action[0])
        # gamma_magnitude = 3*np.arctanh(action[1]+1)
        # gamma_phase = 1.1*np.pi*np.arctanh(action[2])


        # Set noise opertors
        relaxationRate = 0.01
        jump_ops = [np.sqrt(relaxationRate)*sigmam()]      # for the decay

        # Hamiltonian with controls
        H = self.hamiltonian(self.delta, alpha, gamma_magnitude, gamma_phase)
        self.H_array.append(H)              # Array of Hs at each Haar wavelet

        # H_tot for adding Hs at each time bins
        self.H_tot = []

        for ii, H_elem in enumerate(self.H_array):      
            for jj in range(0, num_time_bins):
                Haar_num = self.current_Haar_num - np.floor(ii/self.steps_per_Haar)     # Haar_num: label which Haar wavelet, current_Haar_num: order in the array
                factor = (-1) ** np.floor(jj / (2 ** (Haar_num-1)))                     # factor flips the sign every 2^(Haar_num-1)
                if ii > 0:                 
                    self.H_tot[jj] += factor * H_elem 
                else:                                                                   # Because H_tot[jj] does not exist
                    self.H_tot.append(factor * H_elem)

        
        self.L = []             # at every step we calculate L again because minimal time bin changes
        self.U = np.eye(4)      # identity

        for jj in range(0, num_time_bins):
            L = (liouvillian(Qobj(self.H_tot[jj]), jump_ops, data_only=False, chi=None)).data.toarray()     # Liouvillian calc
            self.L_array.append(L)
            Ut = la.expm(self.final_time/num_time_bins * L)     # time evolution (propagation operator)
            self.U = Ut @ self.U                                # calculate total propagation until the time we are at

        self.state = self.unitary_to_observation(self.U)        # fidelity and flattening -> magnitude, phase

        # Reward and fidelity calculation 
        fidelity = float(np.abs(np.trace(self.U_target.conjugate().transpose()@self.U)))  / (self.U.shape[0])
        reward = (-3*np.log10(1.0-fidelity)+np.log10(1.0-self.prev_fidelity))+(3*fidelity-self.prev_fidelity)
        self.prev_fidelity = fidelity

        # trial to avoid clipping -> failed
        # reward = (-3*np.log10(1.0-fidelity)+np.log10(1.0-self.prev_fidelity))+(fidelity-self.prev_fidelity) + 30*np.log10(1.0000001-abs(gamma_phase/self.gamma_phase_max))**3 + 30*np.log10(1.0000001-abs(2*gamma_magnitude/self.gamma_magnitude_max-1))**3

        # printing on the command line for quick viewing
        print("Step: ",f"{self.current_step_per_Haar:7.3f}","F: ", f"{fidelity:7.3f}","R: ", f"{reward:7.3f}","amp: " f"{action[0]:7.3f}","phase: " f"{action[1]:7.3f}")

        # previous format
        # print("Step: ",f"{self.current_step_per_Haar:7.3f}","F: ", f"{fidelity:7.3f}","R: ", f"{reward:7.3f}","detuning: " f"{action[0]:7.3f}","amp: " f"{action[1]:7.3f}","phase: " f"{action[2]:7.3f}")

        # -----------------------> Refactor <-----------------------
        # GateSynthEnvRLlibHaarNoisy.append_actions(gamma_magnitude, gamma_phase)

        # # append fidelity and reward only at the end of the episode
        # if self.save_data_every_step == 1:
        #     GateSynthEnvRLlibHaarNoisy.append_fidelity(fidelity)
        #     GateSynthEnvRLlibHaarNoisy.append_reward(reward)
        # else:
        #     if self.current_step_per_Haar == self.steps_per_Haar and self.num_Haar_basis == self.current_Haar_num:
        #         GateSynthEnvRLlibHaarNoisy.append_fidelity(fidelity)
        #         GateSynthEnvRLlibHaarNoisy.append_reward(reward)

        # # save the data to .txt file every 1000 episodes
        # if len(GateSynthEnvRLlibHaarNoisy.get_fidelities()) % 1000 == 0:
        #     GateSynthEnvRLlibHaarNoisy.save_data()

        # real time plotting, failed
        # if self.current_Haar_num == self.num_Haar_basis:
        #     GateSynthEnvRLlibHaarNoiseless.scatter_plot.plot(GateSynthEnvRLlibHaarNoiseless.get_fidelities(), GateSynthEnvRLlibHaarNoiseless.get_rewards())
        # ----------------------------------------------------------

        self.transition_history.append([fidelity, reward, *action])

        # Determine if episode is over
        truncated = False
        terminated = False
        if (fidelity >= 1):
            truncated = True                         # truncated when target fidelity reached
        elif (self.current_Haar_num >= self.num_Haar_basis) and (self.current_step_per_Haar >= self.steps_per_Haar): # terminate when all Haar is tested
            terminated = True
        else:
            terminated = False

        if self.current_step_per_Haar == self.steps_per_Haar:           # For each Haar basis, if all trial steps ends, them move to next haar wavelet
            self.current_Haar_num += 1        
            self.current_step_per_Haar = 1
        else:
            self.current_step_per_Haar += 1

        info = {}

        return (self.state, reward, terminated, truncated, info)

    def unitary_to_observation(self, U):
       fidelity = (np.abs(np.trace(self.U_target.conjugate().transpose()@U)))  / (U.shape[0])       #fidelity calculation
       return np.append(fidelity,np.array([(abs(x), (cmath.phase(x)/np.pi+1)/2) for x in U.flatten()], dtype=np.float64).squeeze().reshape(-1)) # cmath phase gives -pi to pi

    def hamiltonian(self, delta, alpha, gamma_magnitude, gamma_phase):
        """Alpha and gamma are complex. This function could be made a callable class attribute."""
        return (delta + alpha)*Z + gamma_magnitude*(np.cos(gamma_phase)*X + np.sin(gamma_phase)*Y)

    @classmethod    
    def append_fidelity(cls,fidelity):      # saving fidelity as the class variable
        cls.fidelities.append(fidelity)

    @classmethod
    def append_reward(cls,reward):          # saving reward as the class variable
        cls.rewards.append(reward)

    @classmethod
    def append_actions(cls, gamma_magnitude, gamma_phase):          # saving actions as the class variable
        cls.gamma_magnitudes.append(gamma_magnitude)
        cls.gamma_phases.append(gamma_phase)

    @classmethod
    def get_fidelities(cls):                # getting array of fidelities
        return cls.fidelities

    @classmethod    
    def get_rewards(cls):                   # getting array of rewards
        return cls.rewards

    @classmethod    
    def get_gamma_magnitudes(cls):                   # getting array of gamma_magnitudes
        return cls.gamma_magnitudes

    @classmethod    
    def get_gamma_phases(cls):                   # getting array of gamma_phases
        return cls.gamma_phases

    @classmethod
    def save_data(cls):                     # save the data to the directory at class variable
        # Get the next file number
        file_num = cls.get_next_file_number()

        # Get the data to be saved
        fidelity_data = cls.get_fidelities()
        reward_data = cls.get_rewards()

        gamma_magnitudes = cls.get_gamma_magnitudes()
        gamma_phases = cls.get_gamma_phases()

        # Create a file name
        file_name = f"data-{file_num:03}.txt"

        # Set the file path
        file_path = os.path.join(cls.data_dir, file_name)

        # Save the data to the file
        with open(file_path, "w") as file:
            for fidelity, reward, mag, phase in zip(fidelity_data, reward_data,gamma_magnitudes, gamma_phases):
                file.write(f"{fidelity},{reward},{mag},{phase}\n")


        # print(f"Data saved to: {file_path}")

    @classmethod
    def get_next_file_number(cls):
        # Get the existing file numbers
        existing_files = []

        if not os.path.exists(cls.data_dir):         # path creation
                    os.mkdir(cls.data_dir)

        for file_name in os.listdir(cls.data_dir):              # getting the last file name
            if file_name.startswith("data-") and file_name.endswith(".txt"):
                file_num = int(file_name[5:-4])
                existing_files.append(file_num)
                os.remove(cls.data_dir+f"data-{file_num:03}.txt")

        # Find the next file number
        if existing_files:
            next_file_num = max(existing_files) + 1
        else:
            next_file_num = 1

        return next_file_num