import numpy as np
from relaqs.api.utils import sample_noise_parameters
from relaqs.environments import SingleQubitEnv, NoisySingleQubitEnv
from relaqs.api.gates import RandomSU2
from qutip.operators import sigmaz, sigmam

noise_file = "april/ibmq_belem_month_is_4.json"
t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)
print(detuning_list)

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
assert noiseless_env.action_space == noisy_env.action_space

n_targets = 100
n_actions = 1000
fidelity_differences = []
for _ in range(n_targets):
    U_target = RandomSU2().get_matrix()
    noiseless_env.U_target = U_target
    noisy_env.U_target = NoisySingleQubitEnv.unitary_to_superoperator(U_target)

    for _ in range(n_actions):
        actions = noiseless_env.action_space.sample()
        noiseless_env.step(actions)
        noisy_env.step(actions)
        fidelity_differences.append(np.abs(noiseless_env.compute_fidelity() - noisy_env.compute_fidelity()))
        noiseless_env.reset()
        noisy_env.reset()

print("Mean fidelity difference: ", np.mean(fidelity_differences))
