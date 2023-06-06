import os
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle('RLlib')
axs[0].set_title('Fidelity')
axs[1].set_title('1 - Fidelity (log scale)')
axs[2].set_title('Reward')

def plot(file_dir):
    # Find the latest file
    max_file_num = -1
    max_file_path = ""
    for file_name in os.listdir(file_dir):
        if file_name.startswith("data-") and file_name.endswith(".txt"):
            file_num = int(file_name[5:-4])
            if file_num > max_file_num:
                max_file_num = file_num
                max_file_path = os.path.join(file_dir, file_name)

    # Read the data from the latest file
    fidelity, reward = [], []
    if max_file_path:
        with open(max_file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                data = line.strip().split(",")
                fidelity.append(float(data[0]))
                reward.append(float(data[1]))

    x = np.arange(len(fidelity))

    axs[0].plot(x, fidelity, c='blue')
    axs[1].semilogy(x, [1 - ii for ii in fidelity], c='red')  # Set y axis to log scale
    axs[2].plot(x, reward, c='green')

    axs[0].set_xlim(0, len(fidelity))
    axs[1].set_xlim(0, len(fidelity))
    axs[2].set_xlim(0, len(reward))

    plt.show()

plot("../../../results/"+os.listdir()[-1]+"/")
