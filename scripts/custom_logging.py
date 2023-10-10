import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data, plot_results
from relaqs.api.callbacks import GateSynthesisCallbacks


def env_creator(config):
    return GateSynthEnvRLlibHaarNoisy(config)

def run(n_training_iterations=1, save=True, plot=True, plot_q_and_gradients=True, figure_title=None):
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

    alg = alg_config.build()
    # ---------------------------------------------------------------------
    list_of_results = []
    # ---------------------> Train Agent <-------------------------
    for _ in range(n_training_iterations):
        result = alg.train()
        list_of_results.append(result['hist_stats'])

    # ---------------------> Save Results <-------------------------
    if save is True:
        env = alg.workers.local_worker().env
        sr = SaveResults(env, alg, results=list_of_results)
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------
    
    assert list_of_results, "train function did not run"
    
    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        plot_data(save_dir, episode_length=alg._episode_history[0].episode_length, figure_title=figure_title)
        print("Plots Created")
    if plot_q_and_gradients:
        assert save is True
        plot_results(save_dir, figure_title=figure_title)
        print("Plots of Average Gradients and Q values Created")
    # --------------------------------------------------------------

if __name__ == "__main__":
    n_training_iterations = 1
    save = True
    plot = True
    plot_q_and_gradients = True
    figure_title ="Noisy environment"
    run(n_training_iterations, save, plot, figure_title=figure_title, plot_q_and_gradients=plot_q_and_gradients)
    