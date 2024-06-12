import numpy as np
from relaqs.environments import SingleQubitEnv, NoisySingleQubitEnv
from relaqs.api.gates import RandomSU2

noiseless_env = SingleQubitEnv(SingleQubitEnv.get_default_env_config())
noisy_env = NoisySingleQubitEnv(NoisySingleQubitEnv.get_default_env_config())
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
