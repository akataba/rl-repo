import os
import sys
sys.path.append('../src')

import json

import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy
import numpy as np

def env_creator(config):
    return GateSynthEnvRLlibHaarNoisy(config)

ray.init()
register_env("my_env", env_creator)
alg_config = DDPGConfig()
alg_config.framework("torch")
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

alg = alg_config.build()
print(type(alg))

###configuration save
data_dir = GateSynthEnvRLlibHaarNoisy.data_dir

if not os.path.exists(data_dir):
    os.mkdir(data_dir)

file_name = "config.txt"
file_path = os.path.join(data_dir, file_name)

ddpg_config = alg_config.to_dict()  # DDPG configuration in dictionary

# for key, value in ddpg_config["env_config"].items():
#     if isinstance(value, np.ndarray):
#         real_list = [c.real for c in value.flatten().tolist()]
#         imaginary_list = [c.imag for c in value.flatten().tolist()]
#         ddpg_config["env_config"][key] = np.array([real_list, imaginary_list]).flatten().tolist()

with open(file_path, "w") as file:
    # DDPG config save
    file.write("DDPG Configuration:\n")
    for key, value in ddpg_config.items():
        file.write(f"{key}: {value}\n")

for _ in range(10000):
    result = alg.train()

path_to_checkpoint = alg.save()

print("path_to_checkpoint", path_to_checkpoint)