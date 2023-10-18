import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
from relaqs.api.callbacks import GateSynthesisCallbacks
import numpy as np
from relaqs.api.callbacks import GateSynthesisCallbacks
from relaqs.api.utils import do_inferencing


def env_creator(config):
    return GateSynthEnvRLlibHaarNoisy(config)

def run(n_training_iterations=1, save=True, plot=True, figure_title=None, inferencing=True, n_episodes_for_inferencing=10):
    ray.init()
    register_env("my_env", env_creator)


    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    alg_config.callbacks(GateSynthesisCallbacks)
    alg_config.environment("my_env", env_config=GateSynthEnvRLlibHaarNoisy.get_default_env_config())

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
    alg_config.target_network_update_freq=5

    # Set H gate
    H = 1/np.sqrt(2) * np.array([[1, 1], [1, -1]])
    env_config = GateSynthEnvRLlibHaarNoisy.get_default_env_config()
    env_config["U_target"] = H


    alg = alg_config.build()
    # ---------------------------------------------------------------------
    list_of_results = []
    # ---------------------> Train Agent <-------------------------
    for _ in range(n_training_iterations):
        result = alg.train()
        list_of_results.append(result['hist_stats'])

    # -----------------------> Inferencing <---------------------------
    if inferencing:
        env, alg = do_inferencing(alg,n_episodes_for_inferencing)
    
    # -------------------> Save Inferencing Results <---------------------------------------
    if save is True:
        sr = SaveResults(env, alg)
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)

    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        plot_data(save_dir, episode_length=alg._episode_history[0].episode_length, figure_title=figure_title)
        print("Plots Created")
    # --------------------------------------------------------------



if __name__ == "__main__":
    n_training_iterations = 5
    save = True
    plot = True
    inferencing=True,
    n_episodes_for_inferencing=3

    figure_title ="Noisy environment"
    run(n_training_iterations, 
        save, 
        plot, 
        figure_title=figure_title, 
        inferencing=inferencing, 
        n_episodes_for_inferencing=n_episodes_for_inferencing
        )
