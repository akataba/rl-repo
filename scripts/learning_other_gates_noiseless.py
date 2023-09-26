""" Learning new single qubit gates, rather than the default X gate. """

import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaar
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
import relaqs.api.gates as gates
import numpy as np

def env_creator(config):
    return GateSynthEnvRLlibHaar(config)

def run(n_training_iterations=1, save=True, plot=True):
    ray.init()
    register_env("my_env", env_creator)
    
    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")

    env_config = GateSynthEnvRLlibHaar.get_default_env_config()

    target_gate = gates.RandomSU2()
    env_config["U_target"] = target_gate.get_matrix()

    alg_config.environment("my_env", env_config=env_config)
    #alg_config.environment(GateSynthEnvRLlibHaar, env_config=GateSynthEnvRLlibHaar.get_default_env_config())

    alg_config.rollouts(batch_mode="complete_episodes")

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
    for _ in range(n_training_iterations):
        result = alg.train()
    # -------------------------------------------------------------

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
    n_training_iterations = 100 
    save = True
    plot = True
    run(n_training_iterations, save, plot)
    
