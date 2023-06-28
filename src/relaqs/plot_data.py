import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np

def plot_data(save_dir, episode_length): 
    """ Currently works for constant episode_length """
    df = pd.read_csv(save_dir + "env_data.csv", header=None)
    fidelities = df.iloc[:, 0]
    rewards = df.iloc[:, 1]

    print("max fidelity", max(fidelities))
    print("max reward", max(rewards))

    final_fidelity_per_episode = []
    final_infelity_per_episode = []
    sum_of_rewards_per_episode = []

    for i in range(episode_length - 1, len(fidelities), episode_length):
        final_fidelity_per_episode.append(fidelities[i])
        final_infelity_per_episode.append(1 - fidelities[i])
        sum_of_rewards_per_episode.append(sum(rewards[i-2 : i+1]))

    rolling_average_window = 100
    avg_final_fidelity_per_episode = []
    avg_final_infelity_per_episode = []
    avg_sum_of_rewards_per_episode = []
    for i in range (len(final_fidelity_per_episode)):
        start = i - rolling_average_window if (i - rolling_average_window) >= 0 else 0
        avg_final_fidelity_per_episode.append(np.mean(final_fidelity_per_episode[start: i]))
        avg_final_infelity_per_episode.append(np.mean(final_infelity_per_episode[start: i]))
        avg_sum_of_rewards_per_episode.append(np.mean(sum_of_rewards_per_episode[start: i]))


    # ----------------> Plotting <----------------
    rcParams['font.family'] = 'serif'

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3) # w/ fidelity plot
    #fig, (ax1, ax2) = plt.subplots(1, 2) # w/o fidelity plot
    fig.suptitle('Noisy Environment')
    fig.set_size_inches(9, 4)

    # ----> fidelity <----
    ax1.plot(final_fidelity_per_episode, color="b", label="a)")
    ax1.plot(avg_final_fidelity_per_episode, color="k")
    ax1.set_title("Fidelity")
    #ax1.set_title("a)", fontfamily='serif', loc='left', fontsize='medium')
    ax1.set_title("a)", loc='left', fontsize='medium')
    #ax1.legend(loc="upper right")
    ax1.set_xlabel("Episodes")
    #ax1.set_xticks([5000, 10000, 15000])
    # --------------------

    # ----> infidelity <----
    ax2.plot(final_infelity_per_episode, color="r")
    ax2.plot(avg_final_infelity_per_episode, color="k")
    ax2.set_yscale("log")
    ax2.set_title("1 - Fidelity (log scale)")
    ax2.set_title("b)", loc='left', fontsize='medium')
    ax2.set_xlabel("Episodes")

    # ax1.plot(final_infelity_per_episode, color="r")
    # ax1.plot(avg_final_infelity_per_episode, color="k")
    # ax1.set_yscale("log")
    # ax1.set_title("1 - Fidelity (log scale)")
    # ax1.set_title("a)", loc='left', fontsize='medium')
    # ax1.set_xlabel("Episodes")
    # ----------------------

    # ----> reward <----
    ax3.plot(sum_of_rewards_per_episode, color="g")
    ax3.plot(avg_sum_of_rewards_per_episode, color="k")
    ax3.set_title("Sum of Rewards")
    ax3.set_title("c)", loc='left', fontsize='medium')
    ax3.set_xlabel("Episodes")

    # ax2.plot(sum_of_rewards_per_episode, color="g")
    # ax2.plot(avg_sum_of_rewards_per_episode, color="k")
    # ax2.set_title("Sum of Rewards")
    # ax2.set_title("b)", loc='left', fontsize='medium')
    # ax2.set_xlabel("Episodes")
    # ------------------

    plt.tight_layout()
    #plt.savefig("plots/noiseless_figure.png")
    #plt.savefig("plots/noisy_infidelity_reward.png")
    #plt.savefig("plots/noiseless_infidelity_reward.png")
    #plt.savefig("plots/9x4_ratio_noiseless.png")
    plt.savefig(save_dir + "plot.png")
    #plt.show()
    # --------------------------------------------