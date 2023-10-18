import sys
sys.path.append('./src/')

import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import TwoQubitGateSynth
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data

from relaqs.quantum_noise_data.get_data import get_month_of_single_qubit_data, get_month_of_all_qubit_data
from relaqs import quantum_noise_data
from relaqs import QUANTUM_NOISE_DATA_DIR
from relaqs import RESULTS_DIR

from qutip.operators import *
from qutip import cnot

import numpy as np
import datetime


def env_creator(config):
    return TwoQubitGateSynth(config)

def save_grad_to_file(resultdict):
    try:
        policydict = resultdict["default_policy"]
        stats = policydict["learner_stats"]
        grad_gnorm = stats["grad_gnorm"]
        with open("gradfile", "a") as f:
            f.write(f"{grad_gnorm}\n")
    except KeyError:
        print(f"Failed to extract grad_gnorm from: {resultdict}")

def inject_logging(alg, logging_func):
    og_ts = alg.training_step
    def new_training_step():
        result = og_ts()
        # do logging here
        logging_func(result)
        return result
    alg.training_step = new_training_step

def run(n_training_iterations=1, save=True, plot=True):
    ray.init()
    try:
        register_env("my_env", env_creator)


        # ---------------------> Configure algorithm and Environment <-------------------------
        alg_config = DDPGConfig()
        alg_config.framework("torch")
        
        env_config = TwoQubitGateSynth.get_default_env_config()
        CNOT = cnot().data.toarray()
        env_config["U_target"] = CNOT

        alg_config.environment("my_env", env_config=env_config)
    
        alg_config.rollouts(batch_mode="complete_episodes")
        alg_config.train_batch_size = TwoQubitGateSynth.get_default_env_config()["steps_per_Haar"]

        ### working 1-3 sets
        alg_config.actor_lr = 4e-5
        alg_config.critic_lr = 5e-4

        alg_config.actor_hidden_activation = "relu"
        alg_config.critic_hidden_activation = "relu"
        alg_config.num_steps_sampled_before_learning_starts = 1000
        alg_config.actor_hiddens = [30,30,30]
        alg_config.exploration_config["scale_timesteps"] = 10000
        print(alg_config.algo_class)
        print(alg_config["framework"])

        alg = alg_config.build()
        inject_logging(alg, save_grad_to_file)
        # ---------------------------------------------------------------------
        list_of_results = []
        # ---------------------> Train Agent <-------------------------
        for _ in range(n_training_iterations):
            result = alg.train()
            list_of_results.append(result['hist_stats'])
        # -------------------------------------------------------------

        # ---------------------> Save Results <-------------------------
        if save is True:
            env = alg.workers.local_worker().env
            sr = SaveResults(env, alg, results=list_of_results, save_path = RESULTS_DIR + "two-qubit gates/"+"CNOT" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/"))
            save_dir = sr.save_results()
            print("Results saved to:", save_dir)
        # --------------------------------------------------------------

        # ---------------------> Plot Data <-------------------------
        if plot is True:
            assert save is True, "If plot=True, then save must also be set to True"
            plot_data(save_dir, episode_length=alg._episode_history[0].episode_length)
            print("Plots Created")
        # --------------------------------------------------------------
    finally:
        ray.shutdown()

if __name__ == "__main__":
    n_training_iterations = 10
    save = True
    plot = True
    run(n_training_iterations, save, plot)
