import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import rv_continuous

class sin_prob_dist(rv_continuous):
    def _pdf(self, theta):
        # The 0.5 is so that the distribution is normalized
        return 0.5 * np.sin(theta)

class Gate(ABC):
    def __init__(self):
        self.sin_sampler = sin_prob_dist(a=0, b=np.pi)

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def get_matrix(self):
        pass

class I(Gate):
    def __str__(self):
        return "I"
    
    def get_matrix(self):
        return np.eye(2)

class X(Gate):
    def __str__(self):
        return "X"
    
    def get_matrix(self):
        return np.array([[0, 1],
                         [1, 0]])

class Y(Gate):
    def __str__(self):
        return "Y"
    
    def get_matrix(self):
        return np.array([[0, -1j],
                         [1j, 0]])

class Z(Gate):
    def __str__(self):
        return "Z"
    
    def get_matrix(self):
        return np.array([[1, 0],
                         [0, -1]])

class H(Gate):
    def __str__(self):
        return "H"
    
    def get_matrix(self):
        return 1/np.sqrt(2) * np.array([[1, 1],
                                        [1, -1]])

class S(Gate):
    def __str__(self):
        return "S"
    
    def get_matrix(self):
        return np.exp(-1j * Z().get_matrix() * np.pi/4)

class X_pi_4(Gate):
    def __str__(self):
        return "X_pi_4"
    
    def get_matrix(self):
        return np.exp(-1j * X().get_matrix() * np.pi/4)

class RandomSU2(Gate):
    def __str__(self):
        return "RandomSU2"
    
    def get_matrix(self):
        """
        Returns a Haar Random element of SU(2).
        https://pennylane.ai/qml/demos/tutorial_haar_measure
        """
        phi = np.random.uniform(low=0, high=2*np.pi)
        omega = np.random.uniform(low=0, high=2*np.pi)
        theta = self.sin_sampler.rvs(size=1)[0]

        U = np.zeros((2, 2), dtype=np.complex128)
        U[0][0] = np.exp(-1j * (phi + omega) / 2) * np.cos(theta / 2)
        U[0][1] = -1 * np.exp(1j * (phi - omega) / 2) * np.sin(theta / 2)
        U[1][0] = np.exp(-1j * (phi - omega) / 2) * np.sin(theta / 2)
        U[1][1] = np.exp(1j * (phi + omega) / 2) * np.cos(theta / 2)

        return U
