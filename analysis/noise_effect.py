import numpy as np
from relaqs.api.utils import sample_noise_parameters
from relaqs.environments import SingleQubitEnv, NoisySingleQubitEnv
from relaqs.api.gates import RandomSU2
from qutip.operators import sigmaz, sigmam
from qiskit.quantum_info import random_statevector, random_unitary  

env_config = NoisySingleQubitEnv.get_default_env_config()
#env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()]
#env_config["delta"] = detuning_list
#env_config["relaxation_ops"] = [sigmam(), sigmaz()]

# zeroing out relaxation rates
env_config["relaxation_rates_list"] = []
env_config["relaxation_ops"] = []
env_config["detuning_list"] = [0]

noiseless_env = SingleQubitEnv(SingleQubitEnv.get_default_env_config())
noisy_env = NoisySingleQubitEnv(env_config)

U_target = RandomSU2().get_matrix()
noiseless_env.U_target = U_target
noisy_env.U_target = NoisySingleQubitEnv.unitary_to_superoperator(U_target)

actions = noiseless_env.action_space.sample()
noiseless_env.step(actions)
noisy_env.step(actions)
noiseless_U = noiseless_env.U
noisy_superop = noisy_env.U

noiseless_superop = NoisySingleQubitEnv.unitary_to_superoperator(noiseless_U)

print("super op close after step:", np.allclose(noiseless_superop, noisy_superop))
