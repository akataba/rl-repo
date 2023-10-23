import os
import sys

try:
    RESULTS_DIR = os.environ["SAVE_BASE_PATH"]
except:
    path_to_relaqs_script = os.path.dirname(os.path.abspath(sys.argv[0]))
    path_to_relaqs_root = os.path.abspath(os.path.join(path_to_relaqs_script, os.pardir))
    RESULTS_DIR = os.path.join(path_to_relaqs_root,"results")


path_to_relaqs_installation = os.path.dirname(os.path.realpath(__file__))
QUANTUM_NOISE_DATA_DIR = path_to_relaqs_installation + "/quantum_noise_data/"


