import os
import ray
from ray import tune
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy, GateSynthEnvRLlibHaar
from ray.tune.search.optuna import OptunaSearch
from relaqs import RESULTS_DIR
import datetime

def env_creator(config):
    return GateSynthEnvRLlibHaarNoisy(config)
    #return GateSynthEnvRLlibHaar(config)

def save_hpt_table(results: tune.ResultGrid):
    df = results.get_dataframe()
    path = RESULTS_DIR + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-HPT/")
    os.makedirs(path)
    df.to_csv(path + "hpt_results.csv")

def run_ray_tune(n_configurations=100, save=True):
    ray.init()
    register_env("my_env", env_creator)
    search_space = {
            "actor_lr" : tune.loguniform(1e-5,1e-4),
            "critic_lr" : tune.loguniform(1e-5,1e-4),
            "actor_num_hiddens" : tune.choice([10, 1e2]),
            "actor_layer_size" : tune.choice([50, 100, 300]),
            "critic_num_hiddens" : tune.choice([10, 1e2]),
            "critic_layer_size" : tune.choice([50, 100, 300])
            }
    algo = OptunaSearch()
    tuner = tune.Tuner(
        objective,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="max_fidelity",
            mode="max",
            search_alg=algo,
            num_samples=n_configurations 
            )
        )
    results = tuner.fit()
    best_fidelity_config = results.get_best_result(metric="max_fidelity", mode="max").config
    print("best_fidelity_config", best_fidelity_config)
    
    # Average within scope
    #best_avg_fidelity_config = results.get_best_result(metric="fidelity", mode="max", scope="last-50-avg").config
    #print("best_avg_fidelity_config", best_avg_fidelity_config)
    
    if save is True:
        save_hpt_table(results)

def objective(config):
    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    env_config = GateSynthEnvRLlibHaarNoisy.get_default_env_config()
    #env_config["U_target"] = gates.X().get_matrix()
    alg_config.environment("my_env", env_config=env_config)

    alg_config.actor_lr = config["actor_lr"]
    alg_config.critic_lr = config["critic_lr"]

    alg_config.actor_hiddens = [config["actor_layer_size"]] * int(config["actor_num_hiddens"])
    alg_config.critic_hiddens = [config["critic_layer_size"]] * int(config["critic_num_hiddens"])

    alg = alg_config.build()
    # ---------------------------------------------------------------------

    # Train
    result = alg.train()

    # Record
    env = alg.workers.local_worker().env
    fidelities = [transition[0] for transition in env.transition_history]
    results = {
            "max_fidelity": max(fidelities),
            "final_fidelity" : fidelities[-1],
            "final_reward" : env.transition_history[-1][1]
        }
    print("\n\n\n", results)
    return results

if __name__ == "__main__":
    n_configurations = 100 
    save = True
    run_ray_tune(n_configurations=n_configurations, save=save)
    ray.shutdown() # not sure if this is required
    