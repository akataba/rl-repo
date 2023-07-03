import os

path_to_relaqs_root = os.path.dirname(os.path.realpath(__file__))
path_relaqs_parent = path_to_relaqs_root.partition("src")[0]
QUANTUM_NOISE_DATA_DIR = path_to_relaqs_root + "/quantum_noise_data/"
RESULTS_DIR = path_relaqs_parent + "results/"
