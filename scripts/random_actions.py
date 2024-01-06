""" Code modified from scripts/two_qubits/random_actions.py on two_qubits branch. """

import sys
sys.path.append('./src/')

from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaar, TwoQubitGateSynth, GateSynthEnvRLlibHaarNoisy
from relaqs import RESULTS_DIR
import numpy as np
import csv

def test_one_qubit_noiseless(n_episodes):
    # Starting with this one as an example
    env = GateSynthEnvRLlibHaar(GateSynthEnvRLlibHaar.get_default_env_config())
    action_space_low = env.action_space.low
    action_space_high = env.action_space.high
    assert len(env.action_space.shape) == 1
    action_space_shape =  env.action_space.shape[0]

    action_history = []
    terminated_history = []
    unitary_history = []
    for _ in range(n_episodes):
        terminated = False
        while terminated is False:
            actions = np.random.uniform(low=action_space_low,
                                        high=action_space_high,
                                        size=action_space_shape)
            action_history.append(actions)
            state, reward, terminated, truncated, info = env.step(actions)
            terminated_history.append(terminated)
            # TODO: add to unitary history  
        env.reset()

    fidelities = [transition[0] for transition in env.transition_history]
    

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

    with open(RESULTS_DIR + "one_qubit_fidelities_random_actions.csv", "w") as f:
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
    n_episodes = 1_000
    test_one_qubit_noiseless(n_episodes)
    #test_one_qubit(n_episodes)
    #test_two_qubits()
