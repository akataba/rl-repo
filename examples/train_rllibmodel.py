import sys
sys.path.append('../src')

from relaqs.api.training import TrainRLLib
from relaqs.environments.gate_synth_env_rllib import * #import all three below
#from relaqs.environments.gate_synth_env_rllib import GateSynthEnvRLlib
#from relaqs.environments.gate_synth_env_rllib import GateSynthEnvRLlibNoiseless
#from relaqs.environments.gate_synth_env_rllib import GateSynthEnvRLlibNoisy

from ray.rllib.algorithms.ddpg import DDPGConfig



# Choose the right one
#trainer = TrainRLLib(DDPGConfig, GateSynthEnvRLlib, episodes=300)
#trainer = TrainRLLib(DDPGConfig, GateSynthEnvRLlibNoiseless, episodes=300)
trainer = TrainRLLib(DDPGConfig, GateSynthEnvRLlibNoisy, episodes=300)

trainer.train_model()
