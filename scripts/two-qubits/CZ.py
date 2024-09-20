import sys
sys.path.append('./src/')

import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments import NoisyTwoQubitEnv
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data


from relaqs import RESULTS_DIR

from qutip.operators import *
from qutip import cphase

import numpy as np
import datetime


def env_creator(config):
    return NoisyTwoQubitEnv(config)

# def save_grad_to_file(resultdict):
#     try:
#         policydict = resultdict["default_policy"]
#         stats = policydict["learner_stats"]
#         grad_gnorm = stats["grad_gnorm"]
#         with open("gradfile", "a") as f:
#             f.write(f"{grad_gnorm}\n")
#     except KeyError:
#         pass
        # print(f"Failed to extract grad_gnorm from: {resultdict}")

def inject_logging(alg, logging_func):
    og_ts = alg.training_step
    def new_training_step():
        result = og_ts()
        # do logging here
        logging_func(result)
        return result
    alg.training_step = new_training_step

def run(n_training_iterations=1, save=True, plot=True):
    ray.init(num_gpus=1)
    # ray.init()
    try:
        register_env("my_env", env_creator)

        # ---------------------> Configure algorithm and Environment <-------------------------
        alg_config = DDPGConfig()
        # alg_config = DDPGConfig()
        alg_config.framework("torch")
        
        env_config = NoisyTwoQubitEnv.get_default_env_config()
        CZ = cphase(np.pi).data.toarray()
        env_config["U_target"] = CZ

        alg_config.environment("my_env", env_config=env_config)
    
        alg_config.rollouts(batch_mode="complete_episodes")
        alg_config.train_batch_size = NoisyTwoQubitEnv.get_default_env_config()["steps_per_Haar"]

        ### working 1-3 sets
        alg_config.actor_lr = 1e-4
        alg_config.critic_lr = 1e-4

        alg_config.actor_hidden_activation = "relu"
        alg_config.critic_hidden_activation = "relu"
        alg_config.num_steps_sampled_before_learning_starts = 5000
        # alg_config.actor_hiddens = [500,20000,500]
        # alg_config.critic_hiddens = [500,20000,500]
        alg_config.actor_hiddens = [1000, 1000, 1000]
        alg_config.critic_hiddens = [1000, 1000, 1000]
        # alg_config.exploration_config["scale_timesteps"] = 200000
        alg_config.exploration_config["scale_timesteps"] = 10000
        
        print(alg_config.algo_class)
        print(alg_config["framework"])

        alg = alg_config.build()
        # inject_logging(alg, save_grad_to_file)
        # ---------------------------------------------------------------------
        list_of_results = []
        # ---------------------> Train Agent <-------------------------
        for ii in range(n_training_iterations):
            result = alg.train()
            list_of_results.append(result['hist_stats'])
            if np.mod(ii,5)==0:
                print("currently",ii,"/",n_training_iterations)
        # -------------------------------------------------------------

        # ---------------------> Save Results <-------------------------
        if save is True:
            env = alg.workers.local_worker().env
            sr = SaveResults(env, alg, results=list_of_results, save_path = RESULTS_DIR + "two-qubit gates/"+"CZ" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/"))
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
    n_training_iterations = 50
    save = True
    plot = True
    run(n_training_iterations, save, plot)
