import ray
from ray.tune.registry import register_env

class TrainRLLib:
    def __init__(self, rllibmodel, env, framework_name="torch", episodes=2) -> None:
        # TODO and docstring and type hints
        ray.init()
        self.env = env
        register_env("my_env", self.env_creator)
        self.alg_config = rllibmodel()
        self.alg_config.framework(framework_name)
        self.alg_config.environment("my_env", env_config=env.get_default_env_config())
        self.alg = self.alg_config.build()
        self.episodes = episodes

    def env_creator(self, config):
        return self.env(config)

    def train_model(self):
        for _ in range(self.episodes):
            result = self.alg.train()


