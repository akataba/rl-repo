""" For refactor of HaarBasis branch, based off of run_and_save_v2 """
import ray
import gymnasium as gym
from ray.rllib.algorithms.ddpg import DDPGConfig
from relaqs.environments import SingleQubitEnv, NoisySingleQubitEnv
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
from relaqs.api import gates

def run(env_class: gym.Env = SingleQubitEnv,
        target_gate: gates.Gate = gates.X(),
        n_training_iterations: int = 1,
        save: bool = True,
        plot: bool = True):
    ray.init()

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    env_config = env_class.get_default_env_config()

    # Set target gate
    env_config["U_target"] = target_gate.get_matrix()

    alg_config.environment(env_class, env_config=env_config)

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
        plot_data(save_dir, episode_length=alg._episode_history[0].episode_length, figure_title=str(target_gate) + "Y, noiselesss, gamma/5")
        print("Plots Created")
    # --------------------------------------------------------------

if __name__ == "__main__":
    env_class = NoisySingleQubitEnv
    target_gate = gates.Y()
    n_training_iterations = 50
    save = plot = True
    run(env_class, target_gate, n_training_iterations, save, plot)
    