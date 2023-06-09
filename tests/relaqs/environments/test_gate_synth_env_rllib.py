import numpy as np
import pytest
import sys
sys.path.append('../src')

from relaqs.environments.gate_synth_env_rllib import GateSynthEnvRLlib

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
