from .environments.gate_synth_env_rllib import GateSynthEnvRLlib
from gym.envs.registration import register
register(
    id='SingleQubitGate-v0',
    entry_point = 'relaqs.environment:GateSynthEnvRLlib',
    max_episode_steps = 5000
)