import ray
import numpy as np
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ddpg import DDPGConfig
from relaqs.quantum_noise_data.get_data import get_month_of_single_qubit_data, get_month_of_all_qubit_data
import pandas as pd
from scipy.linalg import sqrtm
from relaqs import RESULTS_DIR
import ast
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib import GateSynthEnvRLlib
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
from relaqs.save_results import SaveResults
from relaqs.api.callbacks import GateSynthesisCallbacks
from relaqs.plot_data import plot_data
import numpy as np
from relaqs import QUANTUM_NOISE_DATA_DIR
from qutip.operators import *

gate_fidelity = lambda U, V: float(np.abs(np.trace(U.conjugate().transpose() @ V))) / (U.shape[0])

def dm_fidelity(rho, sigma):
    assert np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).imag < 1e-8, f"Non-negligable imaginary component {np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).imag}"
    #return np.abs(np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))))**2
    return np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).real**2

def sample_noise_parameters(path_to_file):
    # ---------------------> Get quantum noise data <-------------------------
    t1_list, t2_list = get_month_of_all_qubit_data(path_to_file)        #in seconds
    mean = 0
    std = 0.03
    sample_size = 100
    samples = np.random.normal(mean, std, sample_size)
    samples_list = samples.tolist()
    return t1_list, t2_list, samples_list

def do_inferencing(alg, n_episodes_for_inferencing, quantum_noise_file_path):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """
    
    assert n_episodes_for_inferencing > 0
    env = alg.workers.local_worker().env
    obs, info = env.reset()
    t1_list, t2_list, detuning_list = sample_noise_parameters(quantum_noise_file_path)
    env.relaxation_rates_list = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()]
    env.delta = detuning_list  
    num_episodes = 0
    episode_reward = 0.0
    print("Inferencing is starting ....")
    while num_episodes < n_episodes_for_inferencing:
        print("episode : ", num_episodes)
        # Compute an action (`a`).
        a = alg.compute_single_action(
            observation=obs,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `a` to the env.
        obs, reward, done, truncated, _ = env.step(a)
        episode_reward += reward
        # Is the episode `done`? -> Reset.
        if done:
            print(f"Episode done: Total reward = {episode_reward}")
            obs, info = env.reset()
            num_episodes += 1
            episode_reward = 0.0
    return env, alg

def load_model(path):
    "path (str): Path to the file usually beginning with the word 'checkpoint' " 
    loaded_model = Algorithm.from_checkpoint(path)
    return loaded_model

def get_best_episode_information(filename):
    df = pd.read_csv(filename, names=['Fidelity', 'Reward', 'Actions', 'Flattened U', 'Episode Id'], header=0)
    fidelity = df.iloc[:, 0]
    max_fidelity_idx = fidelity.argmax()
    fidelity = df.iloc[max_fidelity_idx, 0]
    episode = df.iloc[max_fidelity_idx, 4]
    best_episodes = df[df["Episode Id"] == episode]
    return best_episodes

def env_creator(config):
    return GateSynthEnvRLlibHaarNoisy(config)

def run(gate, n_training_iterations=1, save=True, plot=True,figure_title="",inferencing=True, n_episodes_for_inferencing=10):
    ray.init()
    register_env("my_env", env_creator)
    env_config = GateSynthEnvRLlibHaarNoisy.get_default_env_config()
    env_config["U_target"] = gate

    # ---------------------> Get quantum noise data <-------------------------
    noise_file = QUANTUM_NOISE_DATA_DIR + "april/ibmq_belem_month_is_4.json"
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)

    env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()] # using real T1 data
    env_config["delta"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(),sigmaz()]
    env_config["observation_space_size"] = 2*16 + 1 + 2 + 1 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate + 1 for detuning
    env_config["verbose"] = True

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    alg_config.environment("my_env", env_config=env_config)
    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.callbacks(GateSynthesisCallbacks)
    alg_config.train_batch_size = GateSynthEnvRLlibHaarNoisy.get_default_env_config()["steps_per_Haar"]

    ### working 1-3 sets
    alg_config.actor_lr = 4e-5
    alg_config.critic_lr = 5e-4

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.num_steps_sampled_before_learning_starts = 1000
    alg_config.actor_hiddens = [30,30,30]
    alg_config.exploration_config["scale_timesteps"] = 10000

    alg = alg_config.build()
    list_of_results = []
    for _ in range(n_training_iterations):
        result = alg.train()
        list_of_results.append(result['hist_stats'])

    # loaded_model = load_model(save_result)
    if inferencing:
        env, alg1 = do_inferencing(alg,n_episodes_for_inferencing, "/Users/amara/Dropbox/Zapata/rl_learn/src/relaqs/quantum_noise_data/april/ibmq_manila_month_is_4.json")
    
    if save is True:
        sr = SaveResults(env, alg1)
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)

    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        plot_data(save_dir, episode_length=alg1._episode_history[0].episode_length, figure_title=figure_title)
        print("Plots Created")
    # -------------------------
    ray.shutdown()
    return alg, save_dir





