import os

path_of_current_file = os.path.dirname(os.path.realpath(__file__))
path_to_relaqs_root = path_of_current_file.partition("src")[0]
RESULTS_DIR = path_to_relaqs_root + "results/"
