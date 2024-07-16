""" Example of hyperparameter search over DDPG exploration config"""
import os
import ray
from ray import tune
from ray.air import RunConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from relaqs.environments import NoisySingleQubitEnv
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

    search_space = {"environment" : environment,
                    "n_training_iterations" : n_training_iterations,
                    "random_timesteps": tune.uniform(100, 10_000),
                    "ou_base_scale": tune.uniform(0.01, 0.5),
                    "ou_theta": tune.uniform(0.05, 0.5),
                    "ou_sigma": tune.uniform(0.05, 0.5),
                    "initial_scale": tune.uniform(0.1, 2.0),
                    "scale_timesteps": tune.uniform(1000, 20_000),
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

    exploration_config_keys = alg_config["exploration_config"].keys()
    for key, value in config.items():
        if key in exploration_config_keys:
            alg_config.exploration_config[key] = value

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
    environment = NoisySingleQubitEnv
    n_configurations = 1
    n_training_iterations = 1
    save = True
    run_ray_tune(environment, n_configurations, n_training_iterations, save)
    ray.shutdown() # not sure if this is required
