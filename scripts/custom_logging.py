""" For refactor of HaarBasis branch, based off of run_and_save_v2 """

import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
from ray import tune
import pprint

from typing import Dict, Tuple
import argparse
import numpy as np
import os

import ray
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
import torch
from  torch.linalg import vector_norm
from ray.rllib.utils.torch_utils import global_norm

class GateSynthesisCallbacks(DefaultCallbacks):
    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[str, Policy],
        episode: Episode,
        env_index: int,
        **kwargs
    ):

        print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
        episode.hist_data["q_values"]= []
        episode.hist_data["grad_gnorm"] = []
        episode.hist_data["average_gradnorm"] =[]
        
    def on_postprocess_trajectory(
            self,
            *,
            worker: RolloutWorker,
            episode: Episode,
            agent_id: str,
            policy_id: str,
            policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, Tuple[Policy, SampleBatch]],
            **kwargs
        ):
        print("-------------------post processing batch------------------------------------------------")
        if "num_batches" not in episode.custom_metrics:
            episode.custom_metrics["num_batches"] = 0
        episode.custom_metrics["num_batches"] += 1
        model = worker.get_policy("default_policy").model
        policy = worker.get_policy("default_policy")
        input_dict = SampleBatch(obs=torch.Tensor(postprocessed_batch['obs']))
        #------------------------> getting q values <--------------------------------------------------------
        model_out_t, _ = model(input_dict, [], None)
        q_values = model.get_q_values(model_out_t, torch.Tensor(postprocessed_batch['actions']))
        episode.hist_data["q_values"].append(q_values.detach().numpy()[0])

        #------------------------> getting gradients <--------------------------------------------------------
        batch = SampleBatch(obs=torch.Tensor(postprocessed_batch['obs']),
            actions=torch.Tensor(postprocessed_batch['actions']),
            new_obs = torch.Tensor(postprocessed_batch['new_obs']),
            rewards=torch.Tensor(postprocessed_batch['rewards']),
            terminateds=torch.Tensor(postprocessed_batch['terminateds']),
            truncateds=torch.Tensor(postprocessed_batch['truncateds']),
            weights= torch.Tensor(postprocessed_batch['weights'])
            )
        gradients = policy.compute_gradients(batch)
        gradients_info = gradients[1]
        NoneType = type(None)
        gradients= [x for x in gradients[0] if not isinstance(x, NoneType)]
        average_grad =0
        for grad in gradients:
            average_grad += vector_norm(grad)
        average_grad = average_grad/(len(gradients))
        episode.hist_data['grad_gnorm'].append(gradients_info['learner_stats']['grad_gnorm'])
        episode.hist_data["average_gradnorm"].append(average_grad)

def env_creator(config):
    return GateSynthEnvRLlibHaarNoisy(config)

def run(n_training_iterations=1, save=True, plot=True):
    ray.init()
    register_env("my_env", env_creator)

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    alg_config.callbacks(GateSynthesisCallbacks)
    alg_config.environment("my_env", env_config=GateSynthEnvRLlibHaarNoisy.get_default_env_config())

    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.train_batch_size = GateSynthEnvRLlibHaarNoisy.get_default_env_config()["steps_per_Haar"]

    ### working 1-3 sets
    alg_config.actor_lr = 4e-5
    alg_config.critic_lr = 5e-4

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.num_steps_sampled_before_learning_starts = 1000
    alg_config.actor_hiddens = [30,30,30]
    alg_config.exploration_config["scale_timesteps"] = 10000
    alg_config.target_network_update_freq=5

    alg = alg_config.build()
    # ---------------------------------------------------------------------
    list_of_results = []
    # ---------------------> Train Agent <-------------------------
    for _ in range(n_training_iterations):
        result = alg.train()
        list_of_results.append(result)

    # ---------------------> Save Results <-------------------------
    if save is True:
        env = alg.workers.local_worker().env
        sr = SaveResults(env, alg, results=list_of_results)
        save_dir = sr.save_results()
        print("Results saved to:", save_dir)
    # --------------------------------------------------------------

    # ---------------------> Plot Data <-------------------------
    if plot is True:
        assert save is True, "If plot=True, then save must also be set to True"
        plot_data(save_dir, episode_length=alg._episode_history[0].episode_length)
        print("Plots Created")
    # --------------------------------------------------------------

if __name__ == "__main__":
    n_training_iterations = 1
    save = True
    plot = True
    run(n_training_iterations, save, plot)
    