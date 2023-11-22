import qutip
import matplotlib.pylab as plt
import weylchamber
from weylchamber.visualize import WeylChamber

def plot_weyl_chamber(U=None):
    """ U is a 4x4 unitary and may be a list, numpy array, or qutip operator. """
    w = WeylChamber()
    c1, c2, c3 = weylchamber.c1c2c3(U)
    w.add_point(c1, c2, c3)
    w.plot()
    plt.show()

if __name__ == "__main__":
    # Plot Identity
    gate = qutip.identity([2,2])
    plot_weyl_chamber(gate)

    # Plot CZ
    gate = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]]
    plot_weyl_chamber(gate)