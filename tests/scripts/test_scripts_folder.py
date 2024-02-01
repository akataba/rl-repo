import pytest
import importlib.util
import sys
import os
import ray

def collect_scripts_to_test():
    scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "scripts")
    scripts_to_test = []
    for (root, dirs, files) in os.walk(scripts_dir):
        for file in files:
            if os.path.splitext(file)[1] == ".py":
                scripts_to_test.append(os.path.join(root, file))
    return scripts_to_test


@pytest.fixture(params=collect_scripts_to_test())
def script_path(request):
    return request.param

def test_scripts_dir(script_path):
    #script_path = collect_scripts_to_test()[0]
    spec = importlib.util.spec_from_file_location("relaqs_test_script", script_path)
    module = importlib.util.module_from_spec(spec)

    # If the script is written as a basic script, it will get run upon import
    spec.loader.exec_module(module)

    # If the script is written with a run() function called from if __name__ == "__main__", this code will run it
    # We have to check that this is a run function defined in the script of interest and not the run function from relaqs.api.utils
    if hasattr(module, 'run') and module.__name__ == module.run.__module__:
        module.run()
    ray.shutdown()
