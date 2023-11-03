""" For refactor of HaarBasis branch, based off of run_and_save_v2 """

import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
from relaqs.api import gates
import numpy as np

def env_creator(config):
    return GateSynthEnvRLlibHaarNoisy(config)

def run(n_training_iterations=1, save=True, plot=True):
    ray.init()
    register_env("my_env", env_creator)

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    env_config = GateSynthEnvRLlibHaarNoisy.get_default_env_config()

    # Set target gate
    target_gate = gates.X()
    env_config["U_target"] = target_gate.get_matrix()

    alg_config.environment("my_env", env_config=env_config)
    #alg_config.environment(GateSynthEnvRLlibHaarNoisy, env_config=GateSynthEnvRLlibHaarNoisy.get_default_env_config())

    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = env_config["steps_per_Haar"] # TOOD use env_config

    ### working 1-3 sets
    alg_config.actor_lr = 4e-5
    alg_config.critic_lr = 5e-4

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.num_steps_sampled_before_learning_starts = 1000
    alg_config.actor_hiddens = [30,30,30]
    alg_config.exploration_config["scale_timesteps"] = 10000

    alg = alg_config.build()
    # ---------------------------------------------------------------------

    # ---------------------> Train Agent <-------------------------
    results = [alg.train() for _ in range(n_training_iterations)]
    result = results[-1]
    # ---------------------> Save Results <-------------------------
    if save is True:
        env = alg.workers.local_worker().env
        sr = SaveResults(env, alg, target_gate_string=str(target_gate))
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        plot_data(save_dir, episode_length=alg._episode_history[0].episode_length)
        print("Plots Created")
    # --------------------------------------------------------------

if __name__ == "__main__":
    n_training_iterations = 1
    save = True
    plot = True
    run(n_training_iterations, save, plot)
    
