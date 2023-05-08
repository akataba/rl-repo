"""
This QT gym uses Continuous observation but a DISCRETE Action space

"""
import math
import scipy.linalg as la
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
import random
import time
from random import uniform
import itertools
import sys

import matplotlib.pyplot as plt
from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((-1,1))

sig_p = np.array([[0,1],[0,0]])
sig_m = np.array([[0,0],[1,0]])
X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])
Y = np.array([[0, 1j],[-1j, 0]])

# Actions are alpha (2 actions real and imag) and gamma (2 real and imag)
def oneq_hamiltonian(delta, alpha, gamma):
    # Got from QCTRL tutorial
    
    H = alpha*Z + 0.5*(gamma*sig_m + gamma.conjugate()*sig_p) + delta*Z
    
    return H

## Flatten a complex matrix
# b = np.array([(x.real, x.imag) for x in a.flatten()]).squeeze().reshape(1, -1)

class GateSynthEnv(gym.Env):
    """A environment for single qubit gate quantum control
    
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, delta=1, tf=2, dt=0.01):
        """
        
        """      
        self.tf = tf ## Final time for the gates
        self.dt = dt  ## time step
        self.t = 0
        self.delta = delta # detuning
        self.Utarget = X
        self.U = I
        self.right_init = 0
        self.state =  self.U.flatten().tolist()# Initial state
        
        #Add these to the observation
        self.alpha_r = 0
        self.alpha_i = 0
        self.gamma_r = 0
        self.gamma_i = 0


        self.steps_beyond_done = 0
        self.reward = 0.0
        self.fidelity = 0.0

        #Action is the resolution of amplitudes
        self.num_actions = 3
        self.action_space = spaces.MultiDiscrete([self.num_actions,self.num_actions,self.num_actions,self.num_actions])  # -1 0 1

        self.theta = [0.0, 0.0, 0.0, 0.0]  # angle to rotate for action

        # define observation space
        self.obs_detail_names = ['state','left av photon','right av photon']
        obs_space_low = np.array([0.0,0.0,0.0,0.0,0,0,0,0])
        obs_space_high = np.array([1,1,1,1,180,180,180,180] ) 

        self.observation_space = spaces.Box(
            low=obs_space_low, high=obs_space_high,dtype=np.float32)


    def step(self, action):


        done_flag = False
        
        # Angles to tune for the policy function
        self.theta = [random.uniform(0, 180) for _ in range(4)]

               
        #This will likley be modified but give a general framework
        #Initially let's move by 100
        #If it reaches a bound end the episode
        #Penalty factor for taking the amplitudes too high

        alpha_control_real = self.alpha_r + np.sin(self.theta[0])*100*action[0]
        alpha_control_imag = self.alpha_i + np.sin(self.theta[1])*100*action[1]
        gamma_control_real = self.gamma_r + np.sin(self.theta[2])*100*action[2]
        gamma_control_imag = self.gamma_i + np.sin(self.theta[3])*100*action[3]
        alpha_control = alpha_control_real + 1j*alpha_control_imag
        gamma_control = gamma_control_real + 1j*gamma_control_imag
        
        H = oneq_hamiltonian(self.delta, alpha_control, gamma_control)
        Ut = la.expm(-1j*self.dt*H)
        U = Ut @ self.U # What is the purpose of this operation ?
        self.state = self.U.flatten()
        #leaving off conjugate transpose since X yields itself
        fidelity = 0.5*np.trace(self.Utarget@U)*np.trace(self.Utarget@U).conjugate()
        self.U = U
        
        if fidelity >= 0.95:
            # self.plot_amplitudes()
            print('High efficiency')
            done_flag = True
        elif self.t >= self.tf:
            print('final time reached')
            #self.plot_amplitudes()
            done_flag = True
        elif self.alpha_r > self.delta or self.alpha_i > self.delta or self.gamma_r > self.delta or self.gamma_i > self.delta:
            #I think we need to fix this criteria (something more relevant for previous work)
            print('Amplitudes out of bound')
            done_flag = True

        
        self.alpha_r = alpha_control_real
        self.alpha_i = alpha_control_imag
        self.gamma_r = alpha_control_real
        self.gamma_i = alpha_control_imag
        self.amplitudes.append([self.alpha_r, self.alpha_i, self.gamma_r, self.gamma_i, self.fidelity])
        next_obs = self.get_observation()
        
        self.episode += 1
        self.reward = self.compute_reward(fidelity)
        self.fidelity = fidelity
        self.t = self.t + self.dt
        
        return next_obs, self.reward, done_flag, self.state

    def compute_reward(self, fidelity):
        self.reward = fidelity - self.fidelity
        return self.reward

    def get_observation(self):
        obs = self.state
        obs = np.append(obs,self.theta)
        
        return obs

    
    def plot_amplitudes(self):
        
        fig, [[ax1, ax2],[ax3, ax4],[ax5,ax6]] = plt.subplots(3, 2, figsize=(10,9))
        ax1.plot(np.linspace(0, self.t, int(self.t/self.dt)),[x[0] for x in self.amplitudes])
        ax1.set_xlabel(r'$t\ (\mu s)$')
        ax1.set_ylabel(r'$E_{l, real}$')
        
        ax2.plot(np.linspace(0, self.t, int(self.t/self.dt)),[x[1] for x in self.amplitudes])
        ax2.set_xlabel(r'$t\ (\mu s)$')
        ax2.set_ylabel(r'$E_{l, imag}$')
        
        ax3.plot(np.linspace(0, self.t, int(self.t/self.dt)),[x[2] for x in self.amplitudes])
        ax3.set_xlabel(r'$t\ (\mu s)$')
        ax3.set_ylabel(r'$E_{r, real}$')
        
        ax4.plot(np.linspace(0, self.t, int(self.t/self.dt)),[x[3] for x in self.amplitudes])
        ax4.set_xlabel(r'$t\ (\mu s)$')
        ax4.set_ylabel(r'$|E_{r, imag}$')
        
        ax5.plot(np.linspace(0, self.t, int(self.t/self.dt)),[x[4] for x in self.amplitudes])
        ax5.set_xlabel(r'$t\ (\mu s)$')
        ax5.set_ylabel(r'$\eta$')
        
        fig.savefig("DRLPulses_Tf-{}_dt-{}_tm-{}.png".format(self.t, self.dt, time.time()))
        
        return 0

    def is_done(self):
        return self.steps_beyond_done == 100  # stop after 100 steps in same episode

    def reset(self):
        # set all values to zero
        # Return lower bound of observation
        self.state = [0.0j, 0.0, 0.0, 0.0, 0, 0]
        self.reward = 0.0
        self.fidelity = 0
        self.theta = [0.0, 0.0, 0.0, 0.0]
        self.alpha_r = 0
        self.alpha_i = 0
        self.gamma_r = 0
        self.gamma_i = 0
        self.U = I
        self.episode = 0.0
        self.t = 0.0
        self.amplitudes = []
        return np.array([0.0,0.0,0.0,0.0,0,0,0,0])
