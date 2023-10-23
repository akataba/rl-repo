import os
import sys

path_to_relaqs_installation = os.path.dirname(os.path.realpath(__file__))

try:
    QUANTUM_NOISE_DATA_DIR = os.environ["QUANTUM_NOISE_DATA_DIR"]
except:
    QUANTUM_NOISE_DATA_DIR = path_to_relaqs_installation + "/quantum_noise_data/"

try:
    RESULTS_DIR = os.environ["SAVE_BASE_PATH"]
except:
    path_to_relaqs_alias_github_root = os.path.abspath(os.path.join(os.path.join(path_to_relaqs_installation, os.pardir), os.pardir))
    RESULTS_DIR = os.path.join(path_to_relaqs_alias_github_root,"results")




