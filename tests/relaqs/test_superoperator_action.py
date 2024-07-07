import numpy as np
from relaqs.api.gates import RandomSU2
from qiskit.quantum_info import random_statevector

psi = random_statevector(2).data.reshape(-1, 1)
dm = psi @ psi.conj().T
U = RandomSU2().get_matrix()

# State vector evolution
U_psi = U @ psi
dm1 = U_psi @ U_psi.conj().T

# Density matrix evolution
dm2 = U @ dm @ U.conj().T

# Superoperator evolution, with row-order vectorization
superop1 = np.kron(U, U.conj())
dm3 = (superop1 @ dm.reshape(-1, 1)).reshape(2, 2)

# Superoperator evolution, with column-order vectorization
superop2 = np.kron(U.conj(), U)
dm4 = (superop2 @ dm.reshape(-1, 1, order="F")).reshape(2, 2, order="F")

print(np.allclose(dm1, dm2))
print(np.allclose(dm2, dm3))
print(np.allclose(dm3, dm4))
