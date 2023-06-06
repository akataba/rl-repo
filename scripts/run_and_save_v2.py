import sys
sys.path.append('../src')

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
alg_config.train_batch_size = 1
alg_config.actor_lr = 2e-3
alg_config.critic_lr = 5e-5
alg_config.num_steps_sampled_before_learning_starts = 100
alg = alg_config.build()
print(type(alg))

for _ in range(100):
    result = alg.train()

path_to_checkpoint = alg.save()

print("path_to_checkpoint", path_to_checkpoint)