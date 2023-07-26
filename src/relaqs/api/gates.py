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