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
    spec = importlib.util.spec_from_file_location("relaqs_test_script", script_path)
    module = importlib.util.module_from_spec(spec)
    #sys.modules["relaqs_test_script"] = module
    spec.loader.exec_module(module)

    if hasattr(module, 'run'):
        module.run()
    ray.shutdown()
