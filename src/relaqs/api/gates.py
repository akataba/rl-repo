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
        return np.array([[1, 0],
                         [0, 1j]])

class X_pi_4(Gate):
    def __str__(self):
        return "X_pi_4"
    
    def get_matrix(self):
        theta = np.pi / 4
        return np.array([[np.cos(theta/2), -1j*np.sin(theta/2)],
                         [-1j*np.sin(theta/2), np.cos(theta/2)]])

class XY_combination(Gate):
    def __str__(self):
        return "XY_combination"

    def get_matrix(self):
        """
        aX + bY, a^2 + b^2 = 1
        """
        theta = np.random.uniform(0* np.pi, 2 * np.pi)
        a = np.sin(theta)
        b = np.cos(theta)
        return a * X().get_matrix() + b * Y().get_matrix()

class Rx(Gate):
    def __init__(self, theta_range=(0, 2)):
        super().__init__()
        self.theta_min, self.theta_max = theta_range

    def __str__(self):
        return f"Rx({self.theta_min}, {self.theta_max})"

    def get_matrix(self):
        theta = np.random.uniform(self.theta_min* np.pi, self.theta_max * np.pi)
        return np.array([[np.cos(theta / 2), -1j * np.sin(theta / 2)],
                         [-1j * np.sin(theta / 2), np.cos(theta / 2)]])

class Ry(Gate):
    def __init__(self, theta_range=(0, 2)):
        super().__init__()
        self.theta_min, self.theta_max = theta_range

    def __str__(self):
        return f"Ry({self.theta_min}, {self.theta_max})"

    def get_matrix(self):
        theta = np.random.uniform(self.theta_min* np.pi, self.theta_max * np.pi)
        return np.array([[np.cos(theta / 2), -1 * np.sin(theta / 2)],
                         [ np.sin(theta / 2), np.cos(theta / 2)]])

class Rz(Gate):
    def __init__(self, theta_range=(0, 2)):
        super().__init__()
        self.theta_min, self.theta_max = theta_range

    def __str__(self):
        return f"Rz({self.theta_min}, {self.theta_max})"

    def get_matrix(self):
        theta = np.random.uniform(self.theta_min * np.pi, self.theta_max * np.pi)
        return np.array([[np.exp(-1j*(theta/2)), 0],
                         [ 0, np.exp(1j*(theta/2))]])

class ZX_combination(Gate):
    def __str__(self):
        return "ZX_combination"

    def get_matrix(self):
        """
        aZ + bX, where a^2 + b^2 = 1
        """
        theta = np.random.uniform(0* np.pi, 2 * np.pi)
        a = np.sin(theta)
        b = np.cos(theta)
        return a * Z().get_matrix() + b * X().get_matrix()

class HS(Gate):
    def __str__(self):
        return "HS"

    def get_matrix(self):
        """
        Hadamard followed by Phase (HS)
        """
        return np.matmul(H().get_matrix(), S().get_matrix())


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
