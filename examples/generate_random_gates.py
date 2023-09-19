"""
Methods for generating random gates in U(2) and SU(2)
"""

import numpy as np
from scipy.stats import rv_continuous

class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)

sin_sampler = sin_prob_dist(a=0, b=np.pi)

def get_random_U2():
    """
    Returns a uniform random of U(2).
    Uses QR decomposition procedure from: https://case.edu/artsci/math/esmeckes/Meckes_SAMSI_Lecture2.pdf
    """
    complex_rvs = [np.random.normal(0, 1/np.sqrt(2)) + 1j * np.random.normal(0, 1/np.sqrt(2)) for _ in range(4)] # standard complex gaussian random variables
    A = np.array(complex_rvs).reshape(2, 2)
    U, _ = np.linalg.qr(A)
    assert np.allclose(np.eye(2), U @ U.T.conjugate()) # Check unitarity
    return U

def get_random_SU2():
  """
  Returns a random element of SU(2). Not guaranteed to be Haar random.
  From Google Bard.
  """
  theta = np.random.uniform(0, np.pi)
  phi = np.random.uniform(0, 2 * np.pi)
  u = np.cos(theta / 2)
  v = np.sin(theta / 2) * np.exp(1j * phi)
  return np.array([[u, v], [-np.conjugate(v), u]])

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

if __name__ == "__main__":
    u1 = get_random_U2()
    assert np.allclose(np.eye(2), u1 @ u1.T.conjugate()) # Check unitarity
    print("determinant of element from U(2)", np.linalg.det(u1))

    u2 = get_random_SU2()
    assert np.allclose(np.eye(2), u2 @ u2.T.conjugate()) # Check unitarity
    print("determinant of element from SU(2)", np.linalg.det(u2))

    u3 = get_haar_random_SU2()
    assert np.allclose(np.eye(2), u3 @ u3.T.conjugate()) # Check unitarity
    print("determinant of element from SU(2)", np.linalg.det(u3))  