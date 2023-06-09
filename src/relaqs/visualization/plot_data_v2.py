import os
import matplotlib.pyplot as plt
import numpy as np

fig, axs = plt.subplot_mosaic("ABCDG;ABCEH;ABCFI")
fig.set_figheight(5)
fig.set_figwidth(20)
axs["A"].set_title('Fidelity')
axs["B"].set_title('1 - Fidelity (log scale)')
axs["C"].set_title('Reward')
axs["D"].set_title('magnitude')
axs["G"].set_title('phase')

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
    fidelity, reward, mag, phase = [], [], [], []
    if max_file_path:
        with open(max_file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                data = line.strip().split(",")
                fidelity.append(float(data[0]))
                reward.append(float(data[1]))
                mag.append(float(data[2]))
                phase.append(float(data[3]))

    x = np.arange(len(fidelity))

    axs["A"].plot(x, fidelity, c='blue')
    axs["B"].semilogy(x, [1 - ii for ii in fidelity], c='red')  # Set y axis to log scale
    axs["C"].plot(x, reward, c='green')
    axs["D"].plot([mag[3*ii+1] for ii in range(0,int(np.floor(len(mag)/3)))], c='purple')
    axs["E"].plot([mag[3*ii+2] for ii in range(0,int(np.floor(len(mag)/3)))], c='purple')
    axs["F"].plot([mag[3*ii+3] for ii in range(0,int(np.floor(len(mag)/3)))], c='purple')
    axs["G"].plot([phase[3*ii+1] for ii in range(0,int(np.floor(len(mag)/3)))], c='orange')
    axs["H"].plot([phase[3*ii+2] for ii in range(0,int(np.floor(len(mag)/3)))], c='orange')
    axs["I"].plot([phase[3*ii+3] for ii in range(0,int(np.floor(len(mag)/3)))], c='orange')
    
    axs["A"].set_xlim(0, len(fidelity))
    axs["B"].set_xlim(0, len(fidelity))
    axs["C"].set_xlim(0, len(reward))
    axs["D"].set_xlim(0, np.floor(len(mag)/3))
    axs["E"].set_xlim(0, np.floor(len(mag)/3))
    axs["F"].set_xlim(0, np.floor(len(mag)/3))
    axs["G"].set_xlim(0, np.floor(len(mag)/3))
    axs["H"].set_xlim(0, np.floor(len(mag)/3))
    axs["I"].set_xlim(0, np.floor(len(mag)/3))

    plt.show()

plot("../../../results/"+os.listdir("../../../results/")[-1]+"/")
