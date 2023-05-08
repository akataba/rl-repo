from single_qubit_gate import GateSynthEnv
import random
import wandb


# print("observation space " + str(env.observation_space.shape[0]))
# print("action space " + str(env.action_space.n-1))
def old_test():
    """
    """
    env = GateSynthEnv()
    observation = env.reset()
    for _ in range(1000):
        # env.render()
        # print("action sent from main " + str(action))
        # action = random.randint(0, 1)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        # print(observation)
        print("reward: ", reward)
        print(done)
        print(info)
        print("state")
        print(env.state)

        if done:
            observation = env.reset()
    env.close()


# Graph the reward as function of iterations
# Graph the reward output from ddpg

# Test of the GYM 
def new_basictest():
    """
    """
    env = GateSynthEnv()
    print(env.observation_space.shape[0])
    total_reward = 0
    total_steps = 0

    # 1. Start a new Weights & Biases run
    wandb.init(
        project="quantum-transduction-drl",
        entity="quantum-project",
        save_code=True,
        job_type="basic_test",
        tags=["test", "testing"],
        magic=False,
        sync_tensorboard=False,
        monitor_gym=True,
    )

    # 2. Save model inputs and hyperparameters for Weights & Biases and then use the config object from now on
    wconfig = wandb.config

    wconfig.update(
        {
            "fidelity": env.fidelity,
        }
    )

    while True:
        action = env.action_space.sample()
        print(action)
        obs, reward, done, _ = env.step(action)
        total_reward = reward
        total_steps += 1

        if done:
            break

        print("Episode steps %d Total reward is %.4f" % (total_steps, total_reward))

        # Log the test results to Weights & Biases
        wandb.log({"episode_steps": total_steps, "total_reward": total_reward})


old_test()