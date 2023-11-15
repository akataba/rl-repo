import sys
sys.path.append('./src/')

import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data

from relaqs.quantum_noise_data.get_data import get_month_of_single_qubit_data, get_month_of_all_qubit_data
from relaqs import quantum_noise_data
from relaqs import QUANTUM_NOISE_DATA_DIR
from relaqs import RESULTS_DIR

from qutip.operators import *

import numpy as np
import datetime


def env_creator(config):
    return GateSynthEnvRLlibHaarNoisy(config)

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
        
        env_config = GateSynthEnvRLlibHaarNoisy.get_default_env_config()
        H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
        env_config["U_target"] = H

        # ---------------------> Get quantum noise data <-------------------------
        path_to_file = "C:\\Users\\haknamu\\Documents\\PythonWorkSpace\\GateSynth\\rl-repo\\src\\relaqs\\quantum_noise_data\\april\\ibmq_belem_month_is_4.json"
        t1_list, t2_list = get_month_of_all_qubit_data(path_to_file)        #in seconds
        # ------------------------------------------------------------------------

        env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()] # using real T1 data
        env_config["relaxation_ops"] = [sigmam(),sigmaz()]
        env_config["observation_space_size"] = 2*16 + 1 + 2 + 1 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate + 1 for detuning
        alg_config.environment("my_env", env_config=env_config)
    
        alg_config.rollouts(batch_mode="complete_episodes")
        alg_config.train_batch_size = GateSynthEnvRLlibHaarNoisy.get_default_env_config()["steps_per_Haar"]

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
            sr = SaveResults(env, alg, results=list_of_results, save_path = RESULTS_DIR + "QCE2023/"+"04_noisy_H__" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/"))
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
    n_training_iterations = 100
    save = True
    plot = True
    run(n_training_iterations, save, plot)
    
