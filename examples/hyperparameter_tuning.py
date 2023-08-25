""" For refactor of HaarBasis branch, based off of run_and_save_v2 """

import ray
from ray import tune
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy, GateSynthEnvRLlibHaar
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
from relaqs.api.gates import Gate

def env_creator(config):
    #return GateSynthEnvRLlibHaarNoisy(config)
    return GateSynthEnvRLlibHaar(config)

def run_ray_tune(save=True, plot=True):
    ray.init()
    register_env("my_env", env_creator)
    search_space = {
            "actor_lr" : tune.loguniform(1e-5,1e-2),
            "critic_lr" : tune.loguniform(1e-5,1e-2),
            "actor_num_hiddens" : tune.choice([10, 1e2]),
            "actor_layer_size" : tune.choice([50, 100, 300]),
            "critic_num_hiddens" : tune.choice([10, 1e2]),
            "critic_layer_size" : tune.choice([50, 100, 300])
            # "sampled_before_learning" : tune.loguniform(10,10000),
            # "scale_timesteps" : tune.loguniform(100,100000)
            }
    tuner = tune.Tuner(run_one_training_cycle, param_space = search_space)
    results = tuner.fit()
    best_fidelity_config = results.get_best_result(metric="fidelity", mode="max").config
    best_avg_fidelity_config = results.get_best_result(metric="fidelity", mode="max", scope="last-50-avg").config
    print("best_fidelity_config", best_fidelity_config)
    print("best_avg_fidelity_config", best_avg_fidelity_config)
    return
    alg = run_one_training_cycle(best_fidelity_config, full_output=True)
    # ---------------------> Save Results <-------------------------
    if save is True:
        env = alg.workers.local_worker().env
        sr = SaveResults(env, alg, save_base_path="./max_fidelity/")
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        plot_data(save_dir, episode_length=alg._episode_history[0].episode_length)
        print("Plots Created")
    # --------------------------------------------------------------





def run_one_training_cycle(config, full_output=False):
    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    env_config = GateSynthEnvRLlibHaar.get_default_env_config()
    env_config["U_target"] = Gate.H
    alg_config.environment("my_env", env_config=env_config)
    #alg_config.environment(GateSynthEnvRLlibHaarNoisy, env_config=GateSynthEnvRLlibHaarNoisy.get_default_env_config())

    #alg_config.rollouts(batch_mode="complete_episodes")
    #alg_config.train_batch_size = GateSynthEnvRLlibHaar.get_default_env_config()["steps_per_Haar"]

    ### working 1-3 sets
    alg_config.actor_lr = config["actor_lr"]
    alg_config.critic_lr = config["critic_lr"]

    # alg_config.actor_hidden_activation = "relu"
    # alg_config.critic_hidden_activation = "relu"
    #alg_config.num_steps_sampled_before_learning_starts = config["sampled_before_learning"]
    #alg_config.exploration_config["scale_timesteps"] = config["scale_timesteps"]
    alg_config.actor_hiddens = [config["actor_layer_size"]] * config["actor_num_hiddens"]
    alg_config.critic_hiddens = [config["critic_layer_size"]] * config["critic_num_hiddens"]


    alg = alg_config.build()
    # ---------------------------------------------------------------------

    # ---------------------> Train Agent <-------------------------
    for _ in range(n_training_iterations):
        result = alg.train()

    if full_output:
        return alg
    else:
        env = alg.workers.local_worker().env
        with open(f"debugfile.tsv","a") as f:
            f.write(f"DEBUF fidelity {env.transition_history[-1][0]}")
        return {"fidelity" : env.transition_history[-1][0], "reward" : env.transition_history[-1][1]}

if __name__ == "__main__":
    n_training_iterations = 200
    save = True
    plot = True
    run_ray_tune(save, plot)
    
