from relaqs.environments import NoisySingleQubitEnv
from relaqs.api.gates import RandomSU2
from relaqs.api import gates
from qiskit.quantum_info import random_statevector
import numpy as np
#from qiskit.quantum_info import SuperOp

noisy_env = NoisySingleQubitEnv(NoisySingleQubitEnv.get_default_env_config())

psi = random_statevector(2).data.reshape(-1, 1)
dm = psi @ psi.conj().T

U = gates.X().get_matrix() # Gives np.allclose(U_dm, super_dm) = True
#U = RandomSU2().get_matrix() # Gives np.allclose(U_dm, super_dm) = False

# Regular matrix vector multiplication --> dm
U_psi = U @ psi
U_dm = U_psi @ U_psi.conj().T

# Superoperator evoluion
U_super = noisy_env.unitary_to_superoperator(U)


super_dm = (U_super @ dm.reshape(-1, 1)).reshape(2, 2)

print(U_dm)
print(super_dm)
print(np.allclose(U_dm, super_dm))
