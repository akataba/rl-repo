""" For refactor of HaarBasis branch, based off of run_and_save_v2 """

import ray
# from ray.rllib.algorithms.ddpg import DDPGConfig
from rllib_ddpg.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.single_qubit_env import SingleQubitEnv
from relaqs.environments.noisy_single_qubit_env import NoisySingleQubitEnv
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data,plot_results
from relaqs.api import gates
from relaqs.api.callbacks import GateSynthesisCallbacks
from relaqs.api.utils import sample_noise_parameters
import numpy as np
from qutip import *

def run(env_class=SingleQubitEnv, n_training_iterations=1, save=True, plot=True, figure_title=None):

    noise_file = "april/ibmq_belem_month_is_4.json"
    noise_file_2 = "april/ibmq_quito_month_is_4.json"
    path_to_detuning = "qubit_detuning_data.json"

    # replay_buffer_config = {
    #         "type": "MultiAgentPrioritizedReplayBuffer",
    #         "capacity": 50000,
    #         # Specify prioritized replay by supplying a buffer type that supports
    #         # prioritization, for example: MultiAgentPrioritizedReplayBuffer.
    #         "prioritized_replay": DEPRECATED_VALUE,
    #         # Alpha parameter for prioritized replay buffer.
    #         "prioritized_replay_alpha": 0.6,
    #         # Beta parameter for sampling from prioritized replay buffer.
    #         "prioritized_replay_beta": 0.4,
    #         # Epsilon to add to the TD errors when updating priorities.
    #         "prioritized_replay_eps": 1e-6,
    #         # Whether to compute priorities on workers.
    #         "worker_side_prioritization": False,
    #     }
    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    env_config = env_class.get_default_env_config()

    # Set target gate
    target_gate = gates.H()
    env_config["U_target"] = target_gate.get_matrix()
    # --------------------> Quantum Noise Data<---------------------------------------
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)

    env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()] # using real T1 data
    env_config["delta"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(),sigmaz()]
    env_config["observation_space_size"] = 2*16 + 1 + 2 + 1 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate + 1 for detuning    

    alg_config.environment(env_class, env_config=env_config)

    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = env_config["steps_per_Haar"] # TOOD use env_config
    alg_config.target_noise = 2.532602079724711
    alg_config.twin_q = True
    ### working 1-3 sets
    alg_config.actor_lr = 0.0005
    alg_config.critic_lr = 0.0001
    alg_config.exploration(
                exploration_config={
                    "type": "GaussianNoise",
                    "scale_timesteps": 500,
                    "initial_scale": 1.0,
                    "final_scale": 0.001,
                }
    )
    alg_config.replay_buffer_config["prioritized_replay_alpha"]=0.43927148426471374
    alg_config.replay_buffer_config["prioritized_replay_beta"]= 0.46704809886287607   

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.callbacks(GateSynthesisCallbacks)
    # alg_config.num_steps_sampled_before_learning_starts = 1000
    alg_config.actor_hiddens = [150] * 4
    alg_config.critic_hiddens = [200] * 9
    # alg_config.exploration_config["scale_timesteps"] = 500

    alg = alg_config.build()
    # ---------------------------------------------------------------------

    list_of_results = []
    # ---------------------> Train Agent <-------------------------
    for _ in range(n_training_iterations):
        result = alg.train()
        print(result["episodes_this_iter"])
        list_of_results.append(result['hist_stats'])
    # ---------------------> Save Results <-------------------------
    if save is True:
        env = alg.workers.local_worker().env
        sr = SaveResults(env, alg, target_gate_string=str(target_gate), results=list_of_results)
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        plot_data(save_dir, episode_length=alg._episode_history[0].episode_length)
        plot_results(save_dir,figure_title=figure_title)
        print("Plots Created")
    # --------------------------------------------------------------

if __name__ == "__main__":
    env_class = NoisySingleQubitEnv
    n_training_iterations = 250
    figure_title = "Training the noisy H gate"
    save = True
    plot = True
    run(env_class, n_training_iterations, save, plot, figure_title=figure_title)
    
