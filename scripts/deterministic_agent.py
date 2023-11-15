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
    get_best_episode_information,
    return_env_from_alg
)
from relaqs.api.gates import H


n_training_iterations = 250
figure_title ="Inferencing on multiple noisy environments with different detuning noise"
noise_file = "april/ibmq_belem_month_is_4.json"
noise_file_2 = "april/ibmq_quito_month_is_4.json"
path_to_detuning = "qubit_detuning_data.json"

# --------------------------> Training of model <-----------------------------------------------------
alg = run(H(), 
    n_training_iterations, 
    noise_file=noise_file
    )

# ----------------------- Creating the deterministic agent using actions from the best episode -------------------------------
env = return_env_from_alg(alg)
t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file_2, detuning_noise_file=path_to_detuning)
env.relaxation_rates_list = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()]
env.delta = detuning_list 

sr = SaveResults(env, alg)
save_dir = sr.save_results()
print("Results saved to:", save_dir)

best_episode_information = get_best_episode_information(save_dir + "env_data.csv")

actions = [np.asarray(eval(best_episode_information.iloc[0,2])), np.asarray(eval(best_episode_information.iloc[1,2]))]

num_episodes = 0
episode_reward = 0.0

print("------ The deterministic agent ----------------------")
print("Fidelities from best epsiode: ", [best_episode_information.iloc[0,0], best_episode_information.iloc[1,0]])
obs, info = env.reset()
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


