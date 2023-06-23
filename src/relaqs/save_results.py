import datetime
import os
import csv
from relaqs import RESULTS_DIR


# TODO make these class methods

class SaveResults():
    def __init__(self, env=None, alg=None, save_path=None):
        self.env = env
        self.alg = alg
        if save_path is None:
            self.save_path = self.get_new_directory()
        else:
            self.save_path = save_path
    
        # Create directory if it does not exist
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)

    def get_new_directory(self):
        return RESULTS_DIR + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")

    def save_env_transitions(self):
        # TODO add header labels
        with open(self.save_path + "env_data.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerows(self.env.transition_history)

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
        return self.save_path