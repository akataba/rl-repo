from relaqs.api.training import TrainRLLib
from relaqs.environments.gate_synth_env_rllib import GateSynthEnvRLlib
from ray.rllib.algorithms.ddpg import DDPGConfig


trainer = TrainRLLib(DDPGConfig, GateSynthEnvRLlib, episodes=100)
trainer.train_model()
