import numpy as np
import pytest
import sys
sys.path.append('../src')
from relaqs.save_results import SaveResults
from relaqs.environments.gate_synth_env_rllib import GateSynthEnvRLlib
from relaqs.api.utils import (run, 
    return_env_from_alg, 
    load_and_analyze_best_unitary
)
from relaqs.api.gates import H, I
import pandas as pd
from relaqs import RESULTS_DIR


@pytest.fixture()
def config():
    return GateSynthEnvRLlib.get_default_env_config()

@pytest.fixture()
def gate_environment(config):
    return GateSynthEnvRLlib(config)

@pytest.fixture()
def gate_to_train():
    return H()

def test_environment(gate_environment, config):
    
    # reset the environment
    gate_environment.reset()
    con = config
    np.array_equal(gate_environment.U_initial,con["U_initial"])
    assert 0 == gate_environment.t

def test_unitarity(gate_environment): 
    for _ in range(10):
        action = gate_environment.action_space.sample()
        gate_environment.step(action)
    assert np.allclose(gate_environment.U @ gate_environment.U.T.conjugate(), I().get_matrix())

def test_training(gate_to_train):

    n_training_iterations = 200
    noise_file = "april/ibmq_belem_month_is_4.json"

    alg = run(gate_to_train, 
            n_training_iterations=n_training_iterations,
            noise_file=noise_file 
        )
    env = return_env_from_alg(alg)  
    sr = SaveResults(env, alg)
    save_dir = sr.save_results()
    df = pd.read_csv(save_dir + "env_data.csv")
    last_100_rows = df.tail(100)
    fidelities = last_100_rows.iloc[:,0]
    average_fidelity = sum(fidelities)/len(fidelities)
    assert average_fidelity > 0.995 

def test_loading_of_unitary(gate_to_train):
    data_path = RESULTS_DIR + '2023-11-08_11-09-45/env_data.csv' 
    load_and_analyze_best_unitary(data_path, gate_to_train)
   
  


