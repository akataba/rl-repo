import csv
from relaqs import RESULTS_DIR
import matplotlib.pyplot as plt

def get_fidelities(path):
    with open(path) as f:
        # reading the CSV file
        csvFile = csv.reader(f)
    
        fidelities = []
        # displaying the contents of the CSV file
        for lines in csvFile:
            float_lines = [float(x) for x in lines]
            fidelities.extend(float_lines)

    print(len(fidelities))
    print(max(fidelities))
    return fidelities

if __name__ == "__main__":
    one_qubit = True

    if one_qubit is True:
        path_to_random_fidelities = RESULTS_DIR + "one_qubit_fidelities_random_actions.csv"
        n_qubits_string = "One qubit"
    else:
        path_to_random_fidelities = RESULTS_DIR + "two_qubit_fidelities_random_actions.csv"
        n_qubits_string = "Two qubits"

    fidelities = get_fidelities(path_to_random_fidelities)

    # Plot histogram
    n_bins = 20
    plt.hist(fidelities, bins=n_bins, range=(0, 1))
    plt.xlabel("fidelity")
    plt.ylabel("counts")
    #plt.yscale("log")
    #plt.xscale("log")
    plt.title(n_qubits_string + " 800,000 random actions")
    plt.show()