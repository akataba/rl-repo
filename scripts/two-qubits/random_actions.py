from relaqs.environments.gate_synth_env_rllib_Haar import TwoQubitGateSynth, GateSynthEnvRLlibHaarNoisy
from relaqs import RESULTS_DIR
import numpy as np
import csv

def test_one_qubit():
    n_episodes = 400_000 

    env = GateSynthEnvRLlibHaarNoisy(GateSynthEnvRLlibHaarNoisy.get_default_env_config())
    for _ in range(n_episodes):
        terminated = False
        while terminated is False:
            actions = np.random.uniform(low=-1, high=1, size=2)
            state, reward, terminated, truncated, info = env.step(actions)
        env.reset()

    fidelities = [transition[0] for transition in env.transition_history]

    with open(RESULTS_DIR + "one_qubit_fidelities_random_actions.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(fidelities)

def test_two_qubits():
    n_episodes = 100_000 

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
    test_one_qubit()
    #test_two_qubits()
    
