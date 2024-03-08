import numpy as np
from qutip import Qobj, tensor, cphase
import pytest
import sys
sys.path.append('../src')
from relaqs.save_results import SaveResults
from relaqs.environments.noisy_two_qubit_env import NoisyTwoQubitEnv
from relaqs.environments.noisy_single_qubit_env import NoisySingleQubitEnv
from relaqs.environments.single_qubit_env import SingleQubitEnv

from relaqs.api.utils import (return_env_from_alg, 
    run_noisless_one_qubit_experiment,
    run_noisy_one_qubit_experiment,
)
from relaqs.api.gates import H, I, X
import pandas as pd


@pytest.fixture()
def noisy_config():
    return NoisySingleQubitEnv.get_default_env_config()

@pytest.fixture()
def noiseless_config():
    return SingleQubitEnv.get_default_env_config()

@pytest.fixture()
def noisy_gate_environment(noisy_config):
    return NoisySingleQubitEnv(noisy_config)

@pytest.fixture()
def number_of_training_iterations():
    return 250

@pytest.fixture()
def noiseless_gate_environment(noiseless_config):
    return SingleQubitEnv(noiseless_config)

@pytest.fixture()
def gate_to_train(request):
    if request.param == 'x':
        return X()
    elif request.param == 'h':
        return H()

def test_compute_fidelity_one_qubit_noisy(noisy_config):
    one_qubit_config = noisy_config
    one_qubit_config["U_initial"] = I().get_matrix()
    one_qubit_config["U_target"] = I().get_matrix()

    env = NoisySingleQubitEnv(one_qubit_config)

    assert env.compute_fidelity() == 1.0

    env.U_target = env.unitary_to_superoperator(H().get_matrix())
    assert env.compute_fidelity() == 0.0


def test_compute_fidelity_two_qubits():
    two_qubit_config = NoisyTwoQubitEnv.get_default_env_config()
    I = np.array([[1, 0], [0, 1]])
    II = tensor(Qobj(I),Qobj(I)).data.toarray()
    
    two_qubit_config["U_initial"] = II
    two_qubit_config["U_target"] = II

    env = NoisyTwoQubitEnv(two_qubit_config)

    assert env.compute_fidelity() == 1.0

    CZ = cphase(np.pi).data.toarray()
    env.U_target = env.unitary_to_superoperator(CZ)
    assert env.compute_fidelity() == 0.25

    
    
def test_reseting_environment(noiseless_gate_environment, noiseless_config):
    
    # reset the environment
    noiseless_gate_environment.reset()


    # test that we initialized unitary correctly
    assert np.array_equal(noiseless_gate_environment.U_initial, noiseless_config["U_initial"])

    # for the main branch we shall make the X gate the default target gate for testing purposes
    assert (X().get_matrix(), noiseless_gate_environment.U_target)

    # Previous fidelity is initially set to zero
    assert 0 == noiseless_gate_environment.prev_fidelity
    
    # There should be initial one observation
    assert 9 == len(noiseless_gate_environment.state)

    # Check that the initial flattening happened correctly
    assert np.array_equal(np.array([1,0.5, 0, 0.5, 0, 0.5, 1, 0.5]), noiseless_gate_environment.state[1:9])

def test_unitarity(noiseless_gate_environment): 
    for _ in range(10):
        action = noiseless_gate_environment.action_space.sample()
        noiseless_gate_environment.step(action)
    assert np.allclose(noiseless_gate_environment.U @ noiseless_gate_environment.U.T.conjugate(), I().get_matrix())

@pytest.mark.parametrize("gate_to_train", ['x'], indirect=True)
def test_noisy_training(gate_to_train, number_of_training_iterations):

    n_training_iterations = number_of_training_iterations
    noise_file = "april/ibmq_belem_month_is_4.json"

    alg,_ = run_noisy_one_qubit_experiment(gate_to_train, 
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
    assert average_fidelity > 0.85

@pytest.mark.parametrize("gate_to_train", ['x', 'h'], indirect=True)
def test_noiseless_training(gate_to_train, number_of_training_iterations):
    n_train_iterations= number_of_training_iterations
    alg, _ = run_noisless_one_qubit_experiment(gate_to_train,
            n_training_iterations=n_train_iterations
            )
    env = return_env_from_alg(alg)  
    sr = SaveResults(env, alg)
    save_dir = sr.save_results()
    df = pd.read_csv(save_dir + "env_data.csv")
    last_100_rows = df.tail(100)
    fidelities = last_100_rows.iloc[:,0]
    average_fidelity = sum(fidelities)/len(fidelities)
    assert average_fidelity > 0.850

   
  


