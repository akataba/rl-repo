"""
Averaged superoperator fidelity difference between noisy and noiseless envs
over random targets and actions.
"""
import numpy as np
from relaqs.api.utils import sample_noise_parameters
from relaqs.environments import SingleQubitEnv, NoisySingleQubitEnv
from relaqs.api.gates import RandomSU2
from qutip.operators import sigmaz, sigmam


# Define environments
noiseless_env = SingleQubitEnv(SingleQubitEnv.get_default_env_config())

env_config = NoisySingleQubitEnv.get_default_env_config()
noise_file = "april/ibmq_belem_month_is_4.json"
t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)
env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()]
env_config["detuning_list"] = detuning_list
env_config["relaxation_ops"] = [sigmam(), sigmaz()]
noisy_env = NoisySingleQubitEnv(env_config)

n_targets = 1000
abs_superop_fidelity_differences = []
for _ in range(n_targets):
    # Set random U_target
    U_target = RandomSU2().get_matrix()
    noiseless_env.U_target = U_target
    noisy_env.U_target = NoisySingleQubitEnv.unitary_to_superoperator(U_target)

    # Take random actions
    actions = noiseless_env.action_space.sample()
    noiseless_env.step(actions)
    noisy_env.step(actions)

    # Get superoperators
    noiseless_superop = NoisySingleQubitEnv.unitary_to_superoperator(noiseless_env.U)
    noisy_superop = noisy_env.U

    print("super op close after step:", np.allclose(noiseless_superop, noisy_superop))

    # Compute superoperator fidelities
    noisy_fidelity = noisy_env.compute_fidelity()
    noisy_env.U = noiseless_superop
    noiseless_fidelity = noisy_env.compute_fidelity()

    # Compute superoperator fidelity difference
    superop_fidelity_difference = noiseless_fidelity - noisy_fidelity
    abs_superop_fidelity_differences.append(np.abs(superop_fidelity_difference))

    # Reset envs
    noiseless_env.reset()
    noisy_env.reset()

print("Mean fidelity difference: ", np.mean(abs_superop_fidelity_differences))
