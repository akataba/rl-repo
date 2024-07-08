from relaqs.environments import SingleQubitEnv, NoisySingleQubitEnv
from relaqs.api.gates import RandomSU2

# Generate random unitaries
U = RandomSU2().get_matrix()
U_target = RandomSU2().get_matrix()

# Instantiates environments
noiseless_env = SingleQubitEnv(SingleQubitEnv.get_default_env_config())
noisy_env = NoisySingleQubitEnv(NoisySingleQubitEnv.get_default_env_config())

# Set unitaries
noiseless_env.U_target = U_target
noisy_env.U_target = noisy_env.unitary_to_superoperator(U_target)

noiseless_env.U = U
noisy_env.U = noisy_env.unitary_to_superoperator(U)

# Check that fidelities are the same
noiseless_fidelity = noiseless_env.compute_fidelity()
noisy_fidelity = noisy_env.compute_fidelity()

print("noiseless_fidelity", noiseless_fidelity)
print("noisy_fidelity", noisy_fidelity)
print("Difference", noiseless_fidelity - noisy_fidelity)
print("Is the difference close to zero?", abs(noiseless_fidelity - noisy_fidelity) < 1e-10)
