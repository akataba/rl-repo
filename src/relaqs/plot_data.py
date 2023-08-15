import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import json
from itertools import chain
import matplotlib as mpl

def plot_results(save_dir, episode_length, figure_title="Noisy Environment"):
    with open(save_dir + "train_results_data.json") as file:  # q values and gradient vector norms
        results = json.load(file)

    q_values = [r['q_values'] for r in results] 
    average_grad_norm = [r["average_gradnorm"] for r in results]

    # Flatten lists
    q_values = list(chain.from_iterable(q_values))
    average_grad_norm = list(chain.from_iterable(average_grad_norm))

    # q values
    q_series = pd.Series(q_values)
    q_windows = q_series.rolling(rolling_average_window)
    q_moving_averages = q_windows.mean().to_list()
    
    # gradient norms
    grad_norm_series = pd.Series(average_grad_norm)
    grad_norm_windows = grad_norm_series.rolling(rolling_average_window)
    grad_norm_moving_averages = grad_norm_windows.mean().to_list()

    # -------------------->  q values <--------------------------
    ax4.plot(q_values, color="m")
    ax4.plot(q_moving_averages, color="k")
    ax4.set_title("Q Values")
    ax4.set_title("d)", loc='left', fontsize='medium')
    ax4.set_xlabel("Episodes")

    ax5.plot(average_grad_norm, color="slateblue")
    ax5.plot(grad_norm_moving_averages, color="k")
    ax5.set_title("Average Gradient Norms")
    ax5.set_title("e)", loc='left', fontsize='medium')
    ax5.set_xlabel("Episodes")

def plot_data(save_dir, episode_length, figure_title='Noisy Environment'): 
    """ Currently works for constant episode_length """
    #---------------------- Getting data from files  <--------------------------------------

    with open(save_dir + "env_data.npy", "rb") as f:
        df = np.load(f)




    fidelities = df[:, 0]
    rewards = df[:, 1]

    print("max fidelity", max(fidelities))
    print("max reward", max(rewards))

    final_fidelity_per_episode = []
    final_infelity_per_episode = []
    sum_of_rewards_per_episode = []

    # there is probably a numpy way to speed this up
    for i in range(episode_length - 1, len(fidelities), episode_length):
        final_fidelity_per_episode.append(fidelities[i])
        final_infelity_per_episode.append(1 - fidelities[i])
        sum_of_rewards_per_episode.append(np.sum(rewards[i-2 : i+1]))

    # ----------------------> Moving average <--------------------------------------
    # Fidelity
    rolling_average_window = 100
    avg_final_fidelity_per_episode = []
    avg_final_infelity_per_episode = []
    avg_sum_of_rewards_per_episode = []
    for i in range (len(final_fidelity_per_episode)):
        start = i - rolling_average_window if (i - rolling_average_window) >= 0 else 0
        avg_final_fidelity_per_episode.append(np.mean(final_fidelity_per_episode[start: i]))
        avg_final_infelity_per_episode.append(np.mean(final_infelity_per_episode[start: i]))
        avg_sum_of_rewards_per_episode.append(np.mean(sum_of_rewards_per_episode[start: i]))
    

    # -------------------------------> Plotting <-------------------------------------
    rcParams['font.family'] = 'serif'
    mpl.style.use('seaborn-v0_8')

    fig,((ax1, ax2, ax3), (ax4, ax5, _)) = plt.subplots(2, 3) 
    fig.suptitle(figure_title)
    fig.set_size_inches(10, 5)

    # ----> fidelity <----
    ax1.plot(final_fidelity_per_episode, color="b")
    ax1.plot(avg_final_fidelity_per_episode, color="k")
    ax1.set_title("Fidelity")
    ax1.set_title("a)", loc='left', fontsize='medium')
    ax1.set_xlabel("Episodes")


    # ----> infidelity <----
    ax2.plot(final_infelity_per_episode, color="r")
    ax2.plot(avg_final_infelity_per_episode, color="k")
    ax2.set_yscale("log")
    ax2.set_title("1 - Fidelity (log scale)")
    ax2.set_title("b)", loc='left', fontsize='medium')
    ax2.set_xlabel("Episodes")

    # ----> reward <----
    ax3.plot(sum_of_rewards_per_episode, color="g")
    ax3.plot(avg_sum_of_rewards_per_episode, color="k")
    ax3.set_title("Sum of Rewards")
    ax3.set_title("c)", loc='left', fontsize='medium')
    ax3.set_xlabel("Episodes")
    
    try:
        plot_results(save_dir, episode_length, figure_title)
    except:
        pass

    plt.tight_layout()
    plt.savefig(save_dir + "plot.png")
