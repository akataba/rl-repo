import ray
import numpy as np
from ray.rllib.algorithms.ddpg import DDPGConfig
from relaqs.environments.noisy_single_qubit_env import NoisySingleQubitEnv
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
from relaqs.api import gates
from relaqs.api.utils import sample_noise_parameters
from qutip.operators import sigmaz, sigmam

def run(n_training_iterations=1, save=True, plot=True):
    ray.init()

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    env_config = NoisySingleQubitEnv.get_default_env_config()

    # Set noise parameters
    noise_file = "april/ibmq_belem_month_is_4.json"
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)
    env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()]
    env_config["delta"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(), sigmaz()]
    env_config["observation_space_size"] += 1 # for T2

    # Set target gate
    target_gate = gates.X()
    env_config["U_target"] = target_gate.get_matrix()

    alg_config.environment(NoisySingleQubitEnv, env_config=env_config)

    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = env_config["steps_per_Haar"] 

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
    results = [alg.train() for _ in range(n_training_iterations)]
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
        print("epiosde length", alg._episode_history[0].episode_length)
        plot_data(save_dir, episode_length=alg._episode_history[0].episode_length, figure_title="IBM April Belem Noise, noisy X, gamma/5")
        print("Plots Created")
    # --------------------------------------------------------------

if __name__ == "__main__":
    n_training_iterations = 50
    save = True
    plot = True
    run(n_training_iterations, save, plot)
    