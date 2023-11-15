import os
import ray
from ray import tune
from ray.air import RunConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from relaqs.environments.gate_synth_env_rllib_Haar import TwoQubitGateSynth, GateSynthEnvRLlibHaarNoisy
from ray.tune.search.optuna import OptunaSearch
from relaqs import RESULTS_DIR
import datetime
import numpy as np

def save_hpt_table(results: tune.ResultGrid):
    df = results.get_dataframe()
    path = RESULTS_DIR + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-HPT/")
    os.makedirs(path)
    df.to_csv(path + "hpt_results.csv")

def run_ray_tune(environment, n_configurations=100, n_training_iterations=50, save=True):
    ray.init()
    search_space = {
            "actor_lr" : tune.loguniform(1e-5,1e-3),
            "critic_lr" : tune.loguniform(1e-5,1e-3),
            "actor_num_hiddens" : tune.choice([5, 10, 50]),
            "actor_layer_size" : tune.choice([50, 100, 300, 500]),
            "critic_num_hiddens" : tune.choice([5, 10, 50]),
            "critic_layer_size" : tune.choice([50, 100, 300, 500]),
            "target_noise" : tune.uniform(0.0, 4),
            "n_training_iterations" : n_training_iterations, 
            "environment" : environment
            }
    algo = OptunaSearch()
    tuner = tune.Tuner(
        objective,
        param_space=search_space,
        tune_config=tune.TuneConfig(
            metric="avg_final_fidelities",
            mode="max",
            search_alg=algo,
            num_samples=n_configurations
            ),
        run_config=RunConfig(
            stop={"training_iteration": n_training_iterations},
        ),
        )
    results = tuner.fit()
    best_fidelity_config = results.get_best_result(metric="max_fidelity", mode="max").config
    print("best_fidelity_config", best_fidelity_config)
    
    if save is True:
        save_hpt_table(results)

def objective(config):
    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    env_config = config["environment"].get_default_env_config()
    env_config["verbose"] = False
    alg_config.environment(config["environment"], env_config=env_config)

    alg_config.actor_lr = config["actor_lr"]
    alg_config.critic_lr = config["critic_lr"]

    alg_config.actor_hiddens = [config["actor_layer_size"]] * int(config["actor_num_hiddens"])
    alg_config.critic_hiddens = [config["critic_layer_size"]] * int(config["critic_num_hiddens"])
    alg_config["target_noise"] = config["target_noise"]

    alg = alg_config.build()
    # ---------------------------------------------------------------------

    # Train
    results = [alg.train() for _ in range(config["n_training_iterations"])]

    # Record
    env = alg.workers.local_worker().env
    fidelities = [transition[0] for transition in env.transition_history]
    averageing_window = 50 if len(fidelities) >= 50 else len(fidelities)
    avg_final_fidelities = np.mean([fidelities[-averageing_window:]])
    results = {
            "max_fidelity": max(fidelities),
            "avg_final_fidelities" : avg_final_fidelities,
            "final_fidelity" : fidelities[-1],
            "final_reward" : env.transition_history[-1][1]
        }
    return results

if __name__ == "__main__":
    environment = GateSynthEnvRLlibHaarNoisy
    n_configurations = 25
    n_training_iterations = 100
    save = True
    run_ray_tune(environment, n_configurations, n_training_iterations, save)
    ray.shutdown() # not sure if this is required
    