"""
Steps through the environment taking constant actions per step.
E.g., for a 2-step episode, with action space = 3 the actions should have a shape of (2 x 3).

The intended use of this script is to take the best actions from the noiseless environment
and apply them to the noisy environment to serve as a baseline to compare against an
agent trained on the noise environment.
"""
from relaqs.api.utils import sample_noise_parameters
from relaqs.environments import NoisySingleQubitEnv
from relaqs.api import gates
from qutip.operators import sigmaz, sigmam
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
from relaqs.api.utils import get_best_actions

def run_baseline_actions(actions: list,
                         target_gate: gates.Gate = gates.X(),
                         n_episodes: int = 1,
                         steps_per_episode: int = 2,
                         save: bool = True,
                         plot: bool = True,
                         ):
    env_config = NoisySingleQubitEnv.get_default_env_config()
    t1_list, t2_list, detuning_list = sample_noise_parameters()
    env_config["relaxation_rates_list"] = [t1_list, t2_list]
    env_config["relaxation_ops"] = [sigmam(),sigmaz()]
    env_config["detuning_list"] = detuning_list
    env_config["U_target"] = target_gate.get_matrix()
    noisy_env = NoisySingleQubitEnv(env_config)

    for _ in range(n_episodes):
        for step_id in range(steps_per_episode):
            noisy_env.step(actions[step_id])
        noisy_env.reset()

    # ---------------------> Save Results <-------------------------
    if save is True:
        sr = SaveResults(noisy_env, target_gate_string=str(target_gate))
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        plot_data(save_dir, episode_length=steps_per_episode, figure_title=str(target_gate) + " baseline actions, gamma/7")
        print("Plots Created")
    # --------------------------------------------------------------

if __name__ == "__main__":
    target_gate = gates.X()
    n_episodes = 25000
    steps_per_episode = 2
    file = "/Users/collinfarquhar/Code/rl-repo/results/paper_results/noiseless/2024-07-11_09-45-35_X/env_data.csv"
    actions, fidelity = get_best_actions(file)
    run_baseline_actions(actions, target_gate, n_episodes, steps_per_episode)
