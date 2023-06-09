import sys
sys.path.append('../src')

from relaqs.api.training import TrainRLLib

#from relaqs.environments.gate_synth_env_rllib import * #import all 
from relaqs.environments.gate_synth_env_rllib import GateSynthEnvRLlib
#from relaqs.environments.gate_synth_env_rllib import GateSynthEnvRLlibNoisy

#from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaar
from relaqs.environments.gate_synth_env_rllib_Haar import GateSynthEnvRLlibHaarNoisy

from ray.rllib.algorithms.ddpg import DDPGConfig



# Choose the right one
trainer = TrainRLLib(DDPGConfig, GateSynthEnvRLlib, episodes=300)
#trainer = TrainRLLib(DDPGConfig, GateSynthEnvRLlibNoisy, episodes=300)
#trainer = TrainRLLib(DDPGConfig, GateSynthEnvRLlibHaar, episodes=300)
trainer = TrainRLLib(DDPGConfig, GateSynthEnvRLlibHaarNoisy, episodes=300)

trainer.train_model()
