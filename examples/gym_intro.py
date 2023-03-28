import gym # The original implementations of the OpenAI Gym environments

# Improved implementations of RL algorithms based on Stable Baselines
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# Auxiliary functions for display
import matplotlib.pyplot as plt
# matplotlib inline
# from IPython import display


environment_name = 'CartPole-v1'
# environment_name = 'LunarLander-v2'

env = gym.make(environment_name) # Set up the gym environment
score_for_episode = []

episodes = 500

env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])

# Set up the model
model = A2C('MlpPolicy', env, verbose=1) # https://stable-baselines.readthedocs.io/en/master/modules/a2c.html

# Train the model (It can train well sometimes but screwed if not.)
model.learn(total_timesteps=40000)

# Save the model
model.save("a2c_cartpole")

# Initialize the environment
env = gym.make(environment_name)
state = env.reset()
done = False
score = 0 
step = 0
score_for_episode = []

for episode in range(1, episodes+1): 

    # Initialize the environment
    state = env.reset()
    done = False
    score = 0 
    step = 0 
    while not done:
            
        # Implement policy, which samples from distribution π(a|s)
        action, _states = model.predict(state)
            
        # Update the state and reward (Markov process s->s', collecting reward r(a,s))
        state, reward, done, info = env.step(action)
            
        env.render(mode='rgb_array')
        score+=reward
        step+=1
    score_for_episode.append(score)
 
env.close()

plt.xlabel("Episode")
plt.ylabel("Score for episode")
plt.title("Scoring Actor Critic policy")
plt.scatter(list(range(len(score_for_episode))), score_for_episode)
plt.plot(list(range(len(score_for_episode))), score_for_episode)
plt.show()