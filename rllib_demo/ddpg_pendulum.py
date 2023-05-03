import json
from ray.rllib.algorithms.ddpg import DDPGConfig

def run(print_rewards=False, save_file_name=None):
    config = (
        DDPGConfig()
        .environment("Pendulum-v1")
        .rollouts(num_rollout_workers=2)
        .framework("torch")
        #.training(model={"fcnet_hiddens": [64, 64]})
        .evaluation(evaluation_num_workers=1)
    )

    algo = config.build()

    iterations = 80
    episode_rewards = []
    for i in range(iterations):
        #results.append(algo.train())
        result = algo.train()
        n_new_episodes = result["episodes_this_iter"]
        new_episode_rewards = result["hist_stats"]["episode_reward"][-1 * n_new_episodes :]
        episode_rewards += new_episode_rewards

        if print_rewards is True:
            print("n_new_episodes", n_new_episodes)
            print("new_episode_rewards", new_episode_rewards)
            print("episode_rewards", episode_rewards, "\n")

    if save_file_name is not None:
        # Note: info will also be stored in `~/ray_results/`
        with open("results/ddpg_pendulum/" + save_file_name, 'w') as f:
            # indent=2 is not needed but makes the file human-readable if the data is nested
            json.dump(episode_rewards, f, indent=2)

if __name__ == "__main__":
    #save_file_name = "rewards2.json"
    run()