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

def get_random_su2():
    """
    http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf eqn. (19)
    """  
    alpha = np.random.uniform(low=0, high=2*np.pi)
    psi = np.random.uniform(low=0, high=2*np.pi)
    chi = np.random.uniform(low=0, high=2*np.pi)
    zeta = np.random.randn()
    phi = np.arcsin(np.sqrt(zeta))

    U = np.zeros((2, 2))
    U[0][0] = np.exp(1j * psi) * np.cos(phi)
    U[0][1] = np.exp(1j * chi) * np.sin(phi)
    U[1][0] = -1 * np.exp(-1j * chi) * np.sin(phi)
    U[1][1] = np.exp(-1j * psi) * np.cos(phi)
    U = np.exp(1j * alpha) * U # apply global phase

    return U
