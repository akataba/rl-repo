import pandas as pd
from relaqs import RESULTS_DIR
from relaqs.api import Gate, dm_fidelity

import numpy as np
from numpy.linalg import eigvalsh

#df = pd.read_csv('../results/2023-07-21_13-34-31/env_data.csv')
df = pd.read_csv(RESULTS_DIR + '2023-07-21_13-34-31/env_data.csv') # H
#df = pd.read_csv(RESULTS_DIR + '2023-07-25_16-24-58/env_data.csv') # S
#df = pd.read_csv(RESULTS_DIR + '2023-07-25_22-32-31/env_data.csv') # X pi/4
#df = pd.read_csv(RESULTS_DIR + '2023-07-26_13-14-26/env_data.csv') # X 
#df = pd.read_csv(RESULTS_DIR + '2023-07-26_13-47-46/env_data.csv') # Y 
#df = pd.read_csv(RESULTS_DIR + '2023-08-02_09-10-09/env_data.csv') # Noiseless H
#df = pd.read_csv(RESULTS_DIR + '2023-08-02_09-48-04/env_data.csv') # Noiseless X
#df = pd.read_csv(RESULTS_DIR + '2023-08-02_11-24-32/env_data.csv') # H with element-wise reward 

fidelity = df.iloc[:, 0]
reward = df.iloc[:, 1]
actions = df.iloc[:, 2:4]
unitary = df.iloc[:, 4:]

max_fidelity_idx = fidelity.argmax()
max_fidelity = fidelity.iloc[max_fidelity_idx]
best_fidelity_unitary = np.array([complex(x) for x in unitary.iloc[max_fidelity_idx].tolist()]).reshape(4, 4)

print("Max fidelity:", max_fidelity)
print("Max unitary:", best_fidelity_unitary)

zero = np.array([1, 0]).reshape(-1, 1)
zero_dm = zero @ zero.T.conjugate()
zero_dm_flat = zero_dm.reshape(-1, 1)

dm = best_fidelity_unitary @ zero_dm_flat
dm = dm.reshape(2, 2)
print("Density Matrix:\n", dm)

# Check trace = 1
dm_diagonal = np.diagonal(dm)
print("diagonal:", dm_diagonal)
trace = sum(np.diagonal(dm))
print("trace:", trace)

# # Check that all eigenvalues are positive
eigenvalues = eigvalsh(dm)
print("eigenvalues:", eigenvalues)
#assert (0 <= eigenvalues).all()

#U_target = Gate.H
#U_target = Gate.S
#U_target = Gate.X_pi_4
#U_target = Gate.X
U_target = Gate.H
psi = U_target @ zero
true_dm = psi @ psi.T.conjugate()
print("true dm\n:", true_dm)

print("Density matrix fidelity:", dm_fidelity(true_dm, dm))