import numpy as np
import pytest
import sys
sys.path.append('../src')
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib import GateSynthEnvRLlib
from relaqs.api.utils import get_best_episode_information, run
from relaqs.api.gates import Gate
import pandas as pd

X = np.array([[0,1],[1,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])
Y = np.array([[0, 1j],[-1j, 0]])

@pytest.fixture()
def config():
    return GateSynthEnvRLlib.get_default_env_config()

@pytest.fixture()
def gate_environment(config):
    return GateSynthEnvRLlib(config)

@pytest.fixture()
def gate_to_train():
    return Gate.H

def test_environment(gate_environment, config):
    # assert 8 == len(gate_environment.state)
    # assert (8,) == gate_environment.observation_space.shape
    # assert (4,) == gate_environment.action_space.shape
    
    # reset the environment
    gate_environment.reset()
    con = config
    np.array_equal(gate_environment.U_initial,con["U_initial"])
    assert 0 == gate_environment.t

def test_unitarity(gate_environment): 
    for _ in range(10):
        action = gate_environment.action_space.sample()
        gate_environment.step(action)
    assert np.allclose(gate_environment.U @ gate_environment.U.T.conjugate(), I)

def test_training(gate_to_train):

    n_training_iterations = 200
    save = True
    plot = False
    figure_title ="Inferencing on multiple noisy environments with different detuning noise"
    inferencing=True
    n_episodes_for_inferencing=3

    _ , dir = run(gate_to_train, n_training_iterations, 
        save, 
        plot, 
        figure_title=figure_title, 
        inferencing=inferencing, 
        n_episodes_for_inferencing=n_episodes_for_inferencing,
        )

    df = pd.read_csv(dir + "env_data.csv")
    last_100_rows = df.tail(100)
    fidelities = last_100_rows.iloc[:,0]
    average_fidelity = sum(fidelities)/len(fidelities)
    assert average_fidelity > 0.98  

   
  


