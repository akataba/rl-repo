import os
import inspect

path_to_relaqs_root = os.path.dirname(os.path.realpath(__file__))
path_relaqs_parent = path_to_relaqs_root.partition("src")[0]
QUANTUM_NOISE_DATA_DIR = os.path.join(path_to_relaqs_root, "quantum_noise_data")
RESULTS_DIR = get_results_dir()

def get_results_dir():
    caller_path = os.path.realpath(inspect.stack()[1][1])
    if "rl-repo" in caller_path:
        root_path = caller_path.split("rl-repo")[-2]
        results_path = os.path.join(root_path, "results")
        if os.path.exists(results_path):
            return results_path
    if "scripts" in caller_path:
        before_scripts = caller_path.split("scripts")[-2]
        results_path = os.path.join(before_scripts, "results")
        if os.path.exists(results_path):
            return results_path
    else:
        results_path = os.path.join(os.path.dirname(caller_path), "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        return results_path
