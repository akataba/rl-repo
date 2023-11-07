import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import json
from itertools import chain
import matplotlib as mpl
from relaqs import RESULTS_DIR

def plot_results(save_dir, figure_title=""):
    with open(save_dir + "train_results_data.json") as file:  # q values and gradient vector norms
        results = json.load(file)

    q_values = [r['q_values'] for r in results] 
    average_grad_norm = [r["average_gradnorm"] for r in results]

    # Flatten lists
    q_values = list(chain.from_iterable(q_values))
    average_grad_norm = list(chain.from_iterable(average_grad_norm))

    # q values
    rolling_average_window = 100
    q_series = pd.Series(q_values)
    q_windows = q_series.rolling(rolling_average_window)
    q_moving_averages = q_windows.mean().to_list()
    
    # gradient norms
    grad_norm_series = pd.Series(average_grad_norm)
    grad_norm_windows = grad_norm_series.rolling(rolling_average_window)
    grad_norm_moving_averages = grad_norm_windows.mean().to_list()

    # -------------------->  q values <--------------------------
    # -------------------------------> Plotting <-------------------------------------
    rcParams['font.family'] = 'serif'
    mpl.style.use('seaborn-v0_8')

    fig,(ax1, ax2,) = plt.subplots(1, 2) 
    fig.suptitle(figure_title)
    fig.set_size_inches(10, 5)

    ax1.plot(q_values, color="m")
    ax1.plot(q_moving_averages, color="k")
    ax1.set_title("Q Values")
    ax1.set_title("a)", loc='left', fontsize='medium')
    ax1.set_xlabel("Episodes")

    ax2.plot(average_grad_norm, color="slateblue")
    ax2.plot(grad_norm_moving_averages, color="k")
    ax2.set_title("Average Gradient Norms")
    ax2.set_title("b)", loc='left', fontsize='medium')
    ax2.set_xlabel("Episodes")
    plt.tight_layout()
    plt.savefig(save_dir + "gradient_and_q_values.png")


def plot_data(save_dir, episode_length, figure_title=''):
    """ Currently works for constant episode_length """
    #---------------------- Getting data from files  <--------------------------------------
    try:
        with open(save_dir + "env_data.npy", "rb") as f:
            df = np.load(f)
            fidelities = df[:, 0]
            rewards = df[:, 1]
    except:
        df = pd.read_csv(save_dir + "env_data.csv", header=None)
        fidelities = np.array(df.iloc[:,0])
        rewards = np.array(df.iloc[:,1])



    print("max fidelity", max(fidelities))
    print("max reward", max(rewards))

    final_fidelity_per_episode = []
    final_infelity_per_episode = []
    sum_of_rewards_per_episode = []

    # there is probably a numpy way to speed this up
    n_transitions = len(fidelities)
    for i in range(episode_length, n_transitions, episode_length):
        episode_start_index = i - episode_length
        episode_end_index = i - 1 # 0-indexing
        final_fidelity_per_episode.append(fidelities[episode_end_index])
        final_infelity_per_episode.append(1 - fidelities[episode_end_index])
        sum_of_rewards_per_episode.append(np.sum(rewards[episode_start_index : episode_end_index + 1])) # +1 to include end of episode

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

    fig,(ax1, ax2, ax3) = plt.subplots(1, 3)
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
    
    plt.tight_layout()
    plt.savefig(save_dir + "plot.png")

if __name__ == "__main__":
    save_dir = RESULTS_DIR + "2023-09-19_14-21-02/"
    plot_data(save_dir, episode_length=2, figure_title="Random Target Gate")
