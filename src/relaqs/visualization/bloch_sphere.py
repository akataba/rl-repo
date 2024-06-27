import qutip
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from relaqs import RESULTS_DIR


def plot_bloch_sphere_state(state):
    """ State can be a state vector or density matrix """
    fig = plt.figure()
    b = qutip.Bloch(fig=fig)
    b.add_states(qutip.Qobj(state)) # need to convert to Qobj
    b.render()
    plt.show()

def load_and_plot_noiseless(file_path):
    df = pd.read_csv(RESULTS_DIR + file_path)
    fidelity = df.iloc[:, 0]
    reward = df.iloc[:, 1]
    actions = df.iloc[:, 2:5]
    unitary = df.iloc[:, 5:]
    
    max_fidelity_idx = fidelity.argmax()
    max_fidelity = fidelity.iloc[max_fidelity_idx]
    best_fidelity_unitary = np.array([complex(x) for x in unitary.iloc[max_fidelity_idx].tolist()]).reshape(2, 2)
    
    print("Max fidelity:", max_fidelity)
    print("Max unitary:", best_fidelity_unitary)
    
    zero = np.array([1, 0]).reshape(-1, 1)
    psi = best_fidelity_unitary @ zero
 
    plot_bloch_sphere_state(psi)

def load_and_plot_noisy(file_path):
    df = pd.read_csv(RESULTS_DIR + file_path)
    fidelity = df.iloc[:, 0]
    reward = df.iloc[:, 1]
    actions = df.iloc[:, 2:4] # Only two actions? Up to date?
    superoperators = df.iloc[:, 4:]

    max_fidelity_idx = fidelity.argmax()
    max_fidelity = fidelity.iloc[max_fidelity_idx]
    best_fidelity_superoperator = np.array([complex(x) for x in superoperators.iloc[max_fidelity_idx].tolist()]).reshape(4, 4)

    print("Max fidelity:", max_fidelity)
    print("Max unitary:", best_fidelity_superoperator)

    zero = np.array([1, 0]).reshape(-1, 1)
    zero_dm = zero @ zero.T.conjugate()
    zero_dm_flat = zero_dm.reshape(-1, 1)

    dm = best_fidelity_superoperator @ zero_dm_flat
    dm = dm.reshape(2, 2)
    
    plot_bloch_sphere_state(dm)

if __name__ == "__main__":
    load_and_plot_noiseless('2023-08-02_09-10-09/env_data.csv') # H noiseless
    load_and_plot_noisy('2023-07-21_13-34-31/env_data.csv') # H

