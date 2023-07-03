""" Demo script utilization multiple relaxation rates """

import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
from relaqs.save_results import SaveResults

def env_creator(config):
    return GateSynthEnvRLlibHaarNoisy(config)

def run(n_training_iterations=1, save=False):
    ray.init()
    register_env("my_env", env_creator)

    # ---------------------> Configure algorithm and Environment <-------------------------
    env_config = GateSynthEnvRLlibHaarNoisy.get_default_env_config()
    env_config["relaxation_rates_list"] = [0.01, 0.001] # setting multiple relaxation rates here
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    alg_config.environment("my_env", env_config=env_config)
    #alg_config.environment(GateSynthEnvRLlibHaarNoisy, env_config=GateSynthEnvRLlibHaarNoisy.get_default_env_config())

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

    alg = alg_config.build()
    # ---------------------------------------------------------------------

    # ---------------------> Train Agent <-------------------------
    for _ in range(n_training_iterations):
        result = alg.train()
    # -------------------------------------------------------------

    # ---------------------> Save Results <-------------------------
    if save is True:
        env = alg.workers.local_worker().env
        sr = SaveResults(env, alg)
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

if __name__ == "__main__":
    n_training_iterations = 1
    save = False
    run(n_training_iterations, save)
    