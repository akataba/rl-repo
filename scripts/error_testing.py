from relaqs.api.gates import Rx,Ry,RandomSU2
from relaqs.api import gates
from relaqs.api.utils import *
import numpy as np
from scipy.linalg import sqrtm


def superoperator_to_unitary(S):
    """
    Recovers the unitary matrix U from its superoperator representation S(U) = U* ⊗ U.

    Args:
        S (np.ndarray): The (N^2 x N^2) superoperator matrix.

    Returns:
        np.ndarray: The original (N x N) unitary matrix U, if extraction is possible.

    Raises:
        ValueError: If the input matrix size is not a perfect square.
    """
    # Determine the size of the original unitary matrix
    N_sq = S.shape[0]
    N = int(np.sqrt(N_sq))

    if N * N != N_sq:
        raise ValueError("The input superoperator matrix must have a valid (N^2 x N^2) shape.")

    # Reshape the superoperator into a (N, N, N, N) tensor
    S_reshaped = S.reshape(N, N, N, N)

    # Attempt to recover U using SVD or eigen-decomposition
    # Find the largest singular vector of S_reshaped
    U_star, _, Vh = np.linalg.svd(S_reshaped.reshape(N_sq, N_sq))  # Full SVD

    # Extract the first column (assuming the dominant singular vector corresponds to U*)
    U_star = U_star[:, 0].reshape(N, N)

    # Compute U by conjugating U*
    U = U_star.conj()

    # Ensure U is unitary by projecting onto the unitary space
    U = sqrtm(U @ U.conj().T) @ U  # Enforce unitarity

    return U

def unitary_to_superoperator(U):
    return np.kron(U.conj(), U)

def main():
    temp = [[-0.19985813 + 0.85112373j, - 0.48240321 - 0.05415046j],
        [0.48240321 - 0.05415046j, - 0.19985813 - 0.85112373j]]
    g = [np.array(temp)]

    for gate in g:
        for _ in range(4):
            # g2 = gate.get_matrix()
            g2 = gate
            print(g2)
            print(f"\n")
            g2 = unitary_to_superoperator(g2)
            print(g2)
            print(f"\n")
            g2 = superoperator_to_unitary(g2)
            print(g2)
            print(f"\n\n\n\n")


if __name__ == '__main__':
    main()