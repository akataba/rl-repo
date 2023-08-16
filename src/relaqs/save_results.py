import datetime
import os
import csv
from relaqs import RESULTS_DIR
from typing import List, Dict
import json
import numpy as np
from types import MappingProxyType

l = frozenset([])
FrozenSetType = type(l)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, MappingProxyType):
            return obj.copy()
        if isinstance(obj, FrozenSetType):
            return list(obj)
        else:
            return obj.__dict__         
        return super(NpEncoder, self).default(obj)

class SaveResults():
    def __init__(self, env=None, alg=None, results:List[Dict]=None, save_path=None, save_base_path=None):
        self.env = env
        self.alg = alg
        if save_path is None:
            self.save_path = self.get_new_directory(save_base_path)
        else:
            self.save_path = save_path
    
        # Create directory if it does not exist
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)
        self.results = results

    def get_new_directory(self, save_base_path=None):
        if save_base_path is None:
            return RESULTS_DIR + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
        else:
            return save_base_path + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")

    def save_env_transitions(self):
        # TODO add header labels
        with open(self.save_path + "env_data.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.env.transition_history)

        with open(self.save_path + "env_data.npy", "wb") as f:
            np.save(f, np.array(self.env.transition_history))

    def save_train_results_data(self):
        with open(self.save_path+'train_results_data.json', 'w') as f:
            json.dump(self.results,f, cls=NpEncoder)

    def save_config(self, config_dict):
        config_path = self.save_path + "config.txt"
        with open(config_path, "w") as file:
            for key, value in config_dict.items():
                file.write(f"{key}: {value}\n")

    def save_model(self):
        save_model_path = self.save_path + "model_checkpoints/"
        self.alg.save(save_model_path)

    def save_results(self):
        if self.env is not None:
            self.save_env_transitions()
        if self.alg is not None:
            self.save_config(self.alg.get_config().to_dict())
            self.save_model()
        if self.results is not None:
            self.save_train_results_data()
        return self.save_path
