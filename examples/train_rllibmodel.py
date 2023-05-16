from relaqs.api.training import TrainRLLib
from relaqs.environments.gate_synth_env_rllib import GateSynthEnvRLlib
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.rllib.algorithms.dqn.dqn import DQNConfig


trainer = TrainRLLib(DDPGConfig, GateSynthEnvRLlib, episodes=500)
trainer.train_model()