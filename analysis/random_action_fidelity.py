import csv
import numpy as np
import matplotlib.pyplot as plt
from relaqs import RESULTS_DIR

gate = "X"
#file_name = RESULTS_DIR + "random_actions_fidelity/" + f"noiseless_{gate}.csv"
file_name = RESULTS_DIR + "random_actions_fidelity/" + f"changing_noise_{gate}.csv"

data_list = []
actions_list = []
fidilities = []
with open(file_name, mode ='r')as file:
    csvFile = csv.reader(file)
    for i, line in enumerate(csvFile):
        if i == 0:
            continue # don't read the header
        data = [float(x) for x in line]
        data_list.append(data)
        actions_list.append(data[0:3])
        fidilities.append(data[3])


plot_histogram = True
if plot_histogram:
    counts, bins = np.histogram(fidilities, bins=100)
    plt.stairs(counts, bins)
    #plt.hist(bins[:-1], bins, weights=counts)

    plt.xlabel("Fidelity")
    plt.ylabel("Counts")
    plt.title(f"10,000 Random Actions on {gate} Gate Fidelity Distribution")
    plt.savefig(RESULTS_DIR + "random_actions_fidelity/" + f"noiseless_{gate}_hist.png")
    plt.show()

plot_covariance = True
if plot_covariance:
    actions_list = np.array(actions_list)
    fidilities = np.array(fidilities)
    gamma_mag = actions_list[:, 0]
    gamma_phase = actions_list[:, 1]
    alpha = actions_list[:, 2]

    gamma_mag_cov = np.abs(np.cov(gamma_mag, fidilities)[0, 1])
    gamma_phase_cov = np.abs(np.cov(gamma_phase, fidilities)[0, 1])
    alpha_cov = np.abs(np.cov(alpha, fidilities)[0, 1])
    plt.bar(["gamma magnitude", "gamma phase", "alpha"], [gamma_mag_cov, gamma_phase_cov, alpha_cov])
    plt.title(f"10,000 Random Actions on {gate} Gate Action Abs Covariance with Fidelity")
    plt.savefig(RESULTS_DIR + "random_actions_fidelity/" + f"noiseless_{gate}_covariance.png")
    plt.show()
