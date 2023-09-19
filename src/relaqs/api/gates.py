import numpy as np
from scipy.stats import rv_continuous

class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)

sin_sampler = sin_prob_dist(a=0, b=np.pi)

class Gate: 
    I = np.eye(2)

    X = np.array([[0, 1],
                [1, 0]])

    Y = np.array([[0, 1j],
                [-1j, 0]])

    Z = np.array([[1, 0],
                [0, -1]])

    H = 1/np.sqrt(2) * np.array([[1, 1],
                                [1, -1]])

    S = np.exp(-1j * Z * np.pi/4)

    X_pi_4 = np.exp(-1j * X * np.pi/4)

def get_haar_random_SU2():
    """
    Returns a Haar Random element of SU(2).
    https://pennylane.ai/qml/demos/tutorial_haar_measure
    """
    phi = np.random.uniform(low=0, high=2*np.pi)
    omega = np.random.uniform(low=0, high=2*np.pi)
    theta = sin_sampler.rvs(size=1)[0]

    U = np.zeros((2, 2), dtype=np.complex128)
    U[0][0] = np.exp(-1j * (phi + omega) / 2) * np.cos(theta / 2)
    U[0][1] = -1 * np.exp(1j * (phi - omega) / 2) * np.sin(theta / 2)
    U[1][0] = np.exp(-1j * (phi - omega) / 2) * np.sin(theta / 2)
    U[1][1] = np.exp(1j * (phi + omega) / 2) * np.cos(theta / 2)

    return U
