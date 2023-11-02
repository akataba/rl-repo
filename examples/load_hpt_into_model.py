import warnings
import pandas
from relaqs import RESULTS_DIR
from ray.rllib.algorithms.ddpg import DDPGConfig
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaar, GateSynthEnvRLlibHaarNoisy, TwoQubitGateSynth
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
import gymnasium as gym

def run(
    env_class: gym.Env,
    csv_path: str,
    n_training_iterations: int,
    save: bool,
    plot: bool):
    df = pandas.read_csv(csv_path)

    # get column names in search space
    hyperparam_names = [column_title for column_title in df.columns if "config" in column_title]

    # Get best performing hyperparameters
    metric = "max_fidelity"
    row_id_with_max_metric = df[metric].idxmax()
    row_with_max_metric = df.iloc[row_id_with_max_metric]
    best_hyperparams = row_with_max_metric[hyperparam_names]

    # Initialize algorithm config
    alg_config = DDPGConfig()

    # ---> Set hyperparams not directly in DDPG config <----
    # Set actor hidden layers
    if "config/actor_layer_size" in hyperparam_names and "config/actor_num_hiddens" in hyperparam_names:
        actor_layer_size = best_hyperparams.pop("config/actor_layer_size")
        actor_num_hiddens = int(best_hyperparams.pop("config/actor_num_hiddens"))
        alg_config.actor_hiddens = [actor_layer_size] * actor_num_hiddens

    # Set critic hidden layers
    if "config/critic_layer_size" in hyperparam_names and "config/critic_num_hiddens" in hyperparam_names:
        critic_layer_size = best_hyperparams.pop("config/critic_layer_size")
        critic_num_hiddens = int(best_hyperparams.pop("config/critic_num_hiddens"))
        alg_config.critic_hiddens = [critic_layer_size] * critic_num_hiddens
    # ------------------------------------------------------

    # Set values in DDPG config
    for name in best_hyperparams.index:
        alg_name = name.replace("config/", "")
        if alg_name not in alg_config.keys():
            warnings.warn(f"Hyperparameter {alg_name} not in algorithm config and cannot bet set.")
            continue
        alg_config[alg_name] = best_hyperparams[name]

    # ---------------------> Configure algorithm and Environment <-------------------------
    #alg_config = DDPGConfig()
    alg_config.framework("torch")
    alg_config.environment(env_class, env_config=env_class.get_default_env_config())

    #alg_config.rollouts(batch_mode="complete_episodes")
    #alg_config.train_batch_size = env_class.get_default_env_config()["steps_per_Haar"]

    alg = alg_config.build()
    # ---------------------------------------------------------------------

    results = [alg.train() for _ in range(n_training_iterations)]

    # Save results
    if save is True:
        env = alg.workers.local_worker().env
        sr = SaveResults(env, alg, target_gate_string="CZ_HPT")
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)

    # Plot results
    if plot is True:
        plot_data(save_dir, episode_length=alg._episode_history[0].episode_length)

if __name__ == "__main__":
    env_class = TwoQubitGateSynth
    csv_path = RESULTS_DIR + "2023-10-22_03-17-55-HPT/hpt_results.csv"
    n_training_iterations = 1
    save = True
    plot = True
    run(env_class, csv_path, n_training_iterations, save, plot)