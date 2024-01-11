""" Code modified from scripts/two_qubits/random_actions.py on two_qubits branch. """
import sys
sys.path.append('./src/')

from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaar, TwoQubitGateSynth, GateSynthEnvRLlibHaarNoisy
from relaqs import RESULTS_DIR
import numpy as np
import math
import csv

def polar_vec_to_complex(vec, return_flat=True):
    """ 
    Convert a vector of polar coordinates to a unitary matrix. 
    
    The vector is of the form: [r1, phi1, r2, phi2, ...]
    
    And the matrix is then: [-1 * r1 * exp(i * phi1 * 2pi),...] """
    # Convert polar coordinates to complex numbers
    complex_data = []
    for i in range(0, len(vec), 2):
        r = vec[i]
        phi = vec[i+1]
        z = -1 * r * np.exp(1j * phi * 2*np.pi) # Not sure why the negative sign is needed, but it matches self.U in environment this way. ¯\_(ツ)_/¯
        complex_data.append(z)

    # Reshape into square matrix
    if not return_flat:
        matrix_dimension = math.isqrt(len(vec))
        complex_data = np.array(complex_data).reshape((matrix_dimension, matrix_dimension))

    return complex_data

    
def test_one_qubit_noiseless(n_episodes):
    # Starting with this one as an example
    env = GateSynthEnvRLlibHaar(GateSynthEnvRLlibHaar.get_default_env_config())
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    assert len(env.action_space.shape) == 1
    action_space_shape =  env.action_space.shape[0]

    #action_history = []
    #terminated_history = []
    unitary_history = []
    for _ in range(n_episodes):
        terminated = False
        while terminated is False:
            actions = np.random.uniform(low=action_space_low,
                                        high=action_space_high,
                                        size=action_space_shape)
            #action_history.append(actions)
            state, reward, terminated, truncated, info = env.step(actions)
            #terminated_history.append(terminated)
            if terminated or truncated:
                unitary_history.append(polar_vec_to_complex(state[-8:]))
        env.reset()
    #np.savetxt(RESULTS_DIR + "random_actions_unitaries_noiseless.csv", unitary_history, delimiter=",")
    np.save(RESULTS_DIR + "random_actions_unitaries_noiseless-1000-episodes.npy", unitary_history)

def test_one_qubit(n_episodes):
    env = GateSynthEnvRLlibHaarNoisy(GateSynthEnvRLlibHaarNoisy.get_default_env_config())

    action_history = []
    for _ in range(n_episodes):
        terminated = False
        while terminated is False:
            actions = np.random.uniform(low=-1, high=1, size=2)
            action_history.append(actions)
            state, reward, terminated, truncated, info = env.step(actions)
        env.reset()

    fidelities = [transition[0] for transition in env.transition_history]

    with open(RESULTS_DIR + "one_qubit_fidelities_random_actions-100-episodes.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(fidelities)

def test_two_qubits():
    n_episodes = 10_000 

    env = TwoQubitGateSynth(TwoQubitGateSynth.get_default_env_config())
    for _ in range(n_episodes):
        terminated = False
        while terminated is False:
            actions = np.random.uniform(low=-1, high=1, size=7)
            state, reward, terminated, truncated, info = env.step(actions)
        env.reset()

    fidelities = [transition[0] for transition in env.transition_history]

    with open(RESULTS_DIR + "two_qubit_fidelities_random_actions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(fidelities)

if __name__ == "__main__":
    n_episodes = 1000
    test_one_qubit_noiseless(n_episodes)
    #test_one_qubit(n_episodes)
    #test_two_qubits()
