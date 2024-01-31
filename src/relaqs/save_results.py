import datetime
import os
import csv
from . import get_results_dir
from typing import List, Dict
import json
import numpy as np
import pandas as pd
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
    def __init__(self,
                 env=None,
                 alg=None,
                 results:List[Dict]=None,
                 save_path=None,
                 save_base_path=None,
                 target_gate_string=None
                ):
        self.env = env
        self.alg = alg
        self.target_gate_string = target_gate_string
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
            save_base_path = get_results_dir()

        path = save_base_path + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")

        if self.target_gate_string is not None:
            path = os.path.join(path, "_"  + self.target_gate_string)

        # for backwards compatability
        if path[-1] not in ['/\\']:
            path += "/"
        return path

    def save_env_transitions(self):
        columns = ['Fidelity', 'Rewards', 'Actions', 'Operator', 'Episode Id']
        df = pd.DataFrame(self.env.transition_history, columns=columns)
        df.to_pickle(os.path.join(self.save_path, "env_data.pkl")) # easier to load than csv
        df.to_csv(os.path.join(self.save_path, "env_data.csv", index=False)) # backup in case pickle doesn't work
    
    def save_train_results_data(self):
        with open(os.path.join(self.save_path, 'train_results_data.json', 'w')) as f:
            json.dump(self.results,f, cls=NpEncoder)

    def save_config(self, config_dict):
        config_path = self.save_path + "config.txt"
        with open(config_path, "w") as file:
            for key, value in config_dict.items():
                file.write(f"{key}: {value}\n")

    def save_model(self):
        save_model_path = os.path.join(self.save_path, "model_checkpoints/")
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
