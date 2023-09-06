import numpy as np

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

def get_random_u2():
    """ 
    Returns a uniform random of U(2).
    Uses QR decomposition procedure from: https://case.edu/artsci/math/esmeckes/Meckes_SAMSI_Lecture2.pdf 
    """
    complex_rvs = [np.random.normal(0, 1/np.sqrt(2)) + 1j * np.random.normal(0, 1/np.sqrt(2)) for _ in range(4)] # standard complex gaussian random variables
    A = np.array(complex_rvs).reshape(2, 2)
    U, _ = np.linalg.qr(A)
    assert np.allclose(np.eye(2), U @ U.T.conjugate()) # Check unitarity
    #print("Det", np.linalg.det(U))
    return U

def get_random_su2():
  """
  Returns a random element of SU(2).
  """
  theta = np.random.uniform(0, np.pi)
  phi = np.random.uniform(0, 2 * np.pi)
  u = np.cos(theta / 2)
  v = np.sin(theta / 2) * np.exp(1j * phi)
  return np.array([[u, v], [-np.conjugate(v), u]])

if __name__ == "__main__":
    u1 = get_random_u2()
    assert np.allclose(np.eye(2), u1 @ u1.T.conjugate()) # Check unitarity
    print("determinant of element from U(2)", np.linalg.det(u1))

    u2 = get_random_su2()
    assert np.allclose(np.eye(2), u2 @ u2.T.conjugate()) # Check unitarity
    print("determinant of element from SU(2)", np.linalg.det(u2))  