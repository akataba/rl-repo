from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
import torch
from  torch.linalg import vector_norm
from typing import Dict, Tuple

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
        worker.env.episode_id = episode.episode_id
        episode.hist_data["q_values"]= []
        episode.hist_data["grad_gnorm"] = []
        episode.hist_data["average_gradnorm"] =[]
        episode.hist_data["actions"]=[]
        
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
        episode.hist_data["q_values"].append(q_values.detach().numpy()[0][0])


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
        episode.hist_data["average_gradnorm"].append(average_grad.numpy())
        
        #----------------------------> Getting actions <-----------------------------------------------------
        episode.hist_data["actions"].append(postprocessed_batch["actions"].tolist())

    # def on_episode_end(self,
    #     worker: RolloutWorker,
    #     base_env: BaseEnv,
    #     policies: Dict[str, Policy],
    #     episode: Episode,
    #     env_index: int,
    #     **kwargs):
    #     episode.custom_metrics["actions"] = episode.user_data
       
