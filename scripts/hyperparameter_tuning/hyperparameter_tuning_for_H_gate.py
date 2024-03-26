import os
from ray import tune
from ray.air import RunConfig
from ray.rllib.algorithms.ddpg import DDPGConfig
from relaqs.environments.noisy_single_qubit_env import NoisySingleQubitEnv
from ray.tune.search.optuna import OptunaSearch
from relaqs import RESULTS_DIR
import datetime
import numpy as np
from qutip import *
from relaqs.api.utils import sample_noise_parameters
from relaqs.api.callbacks import GateSynthesisCallbacks
from ray.tune.registry import register_env

def env_creator(config):
    return NoisySingleQubitEnv(config)

def save_hpt_table(results: tune.ResultGrid):
    df = results.get_dataframe()
    path = RESULTS_DIR + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-HPT/")
    os.makedirs(path)
    df.to_csv(path + "hpt_results.csv")

def run_ray_tune(environment, n_configurations=100, n_training_iterations=50, save=True):
    print("starting ray tune .....")
    register_env("my_env", env_creator)
    config = {
            "actor_lr" : [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "critic_lr" : [1e-3, 5e-4, 1e-4, 5e-5, 1e-5],
            "actor_num_hiddens" : tune.choice([5, 10, 50]),
            "actor_layer_size" : tune.choice([50, 100, 300, 500]),
            "critic_num_hiddens" : tune.choice([5, 10, 50]),
            "critic_layer_size" : tune.choice([50, 100, 300, 500]),
            # "target_noise" : tune.uniform(0.0, 4),
            "n_training_iterations" : n_training_iterations, 
            "environment" : environment
            }
    algo = OptunaSearch()
    print("starting OptunaSearch ...")
    tuner = tune.Tuner(
        objective,
        param_space=config,
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
    print(" In objective function and starting build .........")
    noise_file = "april/ibmq_belem_month_is_4.json"
    alg_config = DDPGConfig()
    alg_config.framework("torch")

    env_config = config["environment"].get_default_env_config()
    env_config["verbose"] = False

    # Set target gate
    target_gate = gates.H()
    env_config["U_target"] = target_gate.get_matrix()
    # --------------------> Quantum Noise Data<---------------------------------------
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)
    env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()] # using real T1 data
    env_config["delta"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(),sigmaz()]
    env_config["observation_space_size"] = 2*16 + 1 + 2 + 1 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate + 1 for detuning   

    alg_config.environment(config["environment"], env_config=env_config)
    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = env_config["steps_per_Haar"] # TOOD use env_config
    # alg_config["target_noise"] = config["target_noise"]

    ### working 1-3 sets
    alg_config.actor_lr = config["actor_lr"]
    alg_config.critic_lr = config["critic_lr"]

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.callbacks(GateSynthesisCallbacks)
    alg_config.num_steps_sampled_before_learning_starts = 1000

    alg = alg_config.build()
    # ---------------------------------------------------------------------
    print("------------------------------------finished building-----------------------------")
    # Train
    results = [alg.train() for _ in range(config["n_training_iterations"])]
    print("---------------------finished training ------------------------------------------")

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
    print("-----------------------finished recording ----------------------------------------")
    return results


if __name__ == "__main__":
    environment = NoisySingleQubitEnv
    n_configurations = 25
    n_training_iterations = 200
    save = True
    run_ray_tune(environment, n_configurations, n_training_iterations, save)