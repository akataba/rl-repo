import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
import numpy as np
from relaqs.api.utils import (run,
    sample_noise_parameters,
    do_inferencing,
    get_best_episode_information
)
from relaqs.api.gates import Gate
import ast

n_training_iterations = 250
save = True
plot = False
figure_title ="Inferencing on multiple noisy environments with different detuning noise"
inferencing=True
n_episodes_for_inferencing=3

alg , dir = run(Gate.H, 
    n_training_iterations, 
    save, 
    plot, 
    figure_title=figure_title, 
    inferencing=inferencing, 
    n_episodes_for_inferencing=n_episodes_for_inferencing,
    )

# ----------------------- Creating the deterministic agent using actions from the best episode -------------------------------
env = alg.workers.local_worker().env
obs, info = env.reset()
t1_list, t2_list, detuning_list = sample_noise_parameters( "/Users/amara/Dropbox/Zapata/rl_learn/src/relaqs/quantum_noise_data/april/ibmq_manila_month_is_4.json")
env.relaxation_rates_list = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()]
env.delta = detuning_list 

best_episode_information = get_best_episode_information(dir + "env_data.csv")

actions = [np.asarray(eval(best_episode_information.iloc[0,2])), np.asarray(eval(best_episode_information.iloc[1,2]))]

num_episodes = 0
episode_reward = 0.0
n_episodes_for_inferencing = 10

print("------ The deterministic agent ----------------------")
print("Fidelities from best epsiode: ", [best_episode_information.iloc[0,0], best_episode_information.iloc[1,0]])
for a in actions:
    # Send the computed action `a` to the env.
    print("action taken : ", a)
    obs, reward, done, truncated, _ = env.step(a)
    episode_reward += reward
    # Is the episode `done`? -> Reset.
    if done:
        print(f"Episode done: Total reward = {episode_reward}")
        obs, info = env.reset()
        num_episodes += 1
        episode_reward = 0.0

        
# # ---------------------> Plot Data Deterministic agent <-------------------------
# sr = SaveResults(env, alg)
# save_dir = sr.save_results()
# print("Results saved to:", save_dir)
# plot_data(save_dir, episode_length=alg._episode_history[0].episode_length, figure_title="Deterministic Agent: Actions leading to max fidelity")
# print(" Deterministic Agent Plots Created")
