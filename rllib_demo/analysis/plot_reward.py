import json
import matplotlib.pyplot as plt

with open("../results/ddpg_pendulum/rewards1.json", 'r') as f:
    rewards = json.load(f)

print(rewards)
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Episode Reward")
plt.title("DDPG on Pendulum-v1")
#plt.savefig("../plots/ddpg_on_pendulum-v1.png")
plt.show()
