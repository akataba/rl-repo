import matplotlib.pyplot as plt
from matplotlib import rcParams
import pandas as pd
import numpy as np
import json
from itertools import chain
import matplotlib as mpl
from relaqs import RESULTS_DIR
from relaqs.api.utils import *
from relaqs.api.gates import Rx,Ry,RandomSU2


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


def plot_data(save_dir, episode_length= None, figure_title='', plot_filename= 'plot', perform_analysis=False):
    """ Currently works for constant episode_length """
    #---------------------- Getting data from files  <--------------------------------------
    df = pd.read_csv(save_dir + "env_data.csv", header=0)
    fidelities = np.array(df.iloc[:,0])
    rewards = np.array(df.iloc[:,1])
    episode_ids = np.array(df.iloc[:,-1])

    if perform_analysis:
        perform_action_analysis(df=df)

    print("max fidelity: ", max(fidelities))
    print("max reward: ", max(rewards))

    # --------> Get fidelity, infidelity, and reward from the last step of the episode <--------
    final_fidelity_per_episode = []
    final_infelity_per_episode = []
    sum_of_rewards_per_episode = []

    current_episode_id = episode_ids[0]
    current_fidelity = fidelities[0]
    current_reward_sum = rewards[0]
    for i in range(len(episode_ids)):
        if (episode_ids[i] != current_episode_id) or (i == len(episode_ids) - 1):
            final_fidelity_per_episode.append(current_fidelity)
            final_infelity_per_episode.append(1 - current_fidelity)
            sum_of_rewards_per_episode.append(current_reward_sum)
            current_reward_sum = 0
        current_episode_id = episode_ids[i]
        current_fidelity = fidelities[i]
        current_reward_sum += rewards[i]
    # ------------------------------------------------------------------------------------------

    # ----------------------> Moving average <--------------------------------------
    # Fidelity
    rolling_average_window = 100
    avg_final_fidelity_per_episode = []
    avg_final_infelity_per_episode = []
    avg_sum_of_rewards_per_episode = []
    for i in range (len(final_fidelity_per_episode)):
        start = i - rolling_average_window if (i - rolling_average_window) > 0 else 0
        avg_final_fidelity_per_episode.append(np.mean(final_fidelity_per_episode[start: i + 1]))
        avg_final_infelity_per_episode.append(np.mean(final_infelity_per_episode[start: i + 1]))
        avg_sum_of_rewards_per_episode.append(np.mean(sum_of_rewards_per_episode[start: i + 1]))

    # Round averages to prevent numerical error when plotting
    rounding_precision = 6
    avg_final_fidelity_per_episode = np.round(avg_final_fidelity_per_episode, rounding_precision)
    avg_final_infelity_per_episode = np.round(avg_final_infelity_per_episode, rounding_precision)
    avg_sum_of_rewards_per_episode = np.round(avg_sum_of_rewards_per_episode, rounding_precision)


    if len(avg_final_fidelity_per_episode) >= 100: 
        print("Average final fidelity over last 100 episodes", np.mean(avg_final_fidelity_per_episode[-100:]))

    # -------------------------------> Plotting <-------------------------------------
    rcParams['font.family'] = 'serif'
    mpl.style.use('seaborn-v0_8')

    fig,(ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(figure_title)
    fig.set_size_inches(10, 5)

    # ----> fidelity <----
    ax1.scatter(range(len(final_fidelity_per_episode)), final_fidelity_per_episode, color="b", s=10)
    ax1.scatter(range(len(avg_final_fidelity_per_episode)), avg_final_fidelity_per_episode, color="k", s=10)
    ax1.set_title("Fidelity")
    ax1.set_title("a)", loc='left', fontsize='medium')
    ax1.set_xlabel("Episodes")

    # ----> infidelity <----
    ax2.scatter(range(len(final_infelity_per_episode)), final_infelity_per_episode, color="r", s=10)
    ax2.scatter(range(len(avg_final_infelity_per_episode)), avg_final_infelity_per_episode, color="k", s=10)
    ax2.set_yscale("log")
    ax2.set_title("1 - Fidelity (log scale)")
    ax2.set_title("b)", loc='left', fontsize='medium')
    ax2.set_xlabel("Episodes")

    # ----> reward <----
    ax3.scatter(range(len(sum_of_rewards_per_episode)), sum_of_rewards_per_episode, color="g", s=10)
    ax3.scatter(range(len(avg_sum_of_rewards_per_episode)), avg_sum_of_rewards_per_episode, color="k", s=10)
    ax3.set_title("Sum of Rewards")
    ax3.set_title("c)", loc='left', fontsize='medium')
    ax3.set_xlabel("Episodes")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, plot_filename))

def inference_plot(fidelities, infidelity, rewards, figure_title, save_dir, plot_filename):
    # -------------------------------> Plotting Section <-------------------------------------
    # Configure plotting styles for consistent visualization aesthetics
    rcParams['font.family'] = 'serif'
    mpl.style.use('seaborn-v0_8')

    # Create a figure with 3 subplots arranged in a row
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.suptitle(figure_title)  # Set overall figure title
    fig.set_size_inches(10, 5)  # Adjust figure size

    # ----> Scatter plot of fidelity per instance <----
    ax1.scatter(range(len(fidelities)), fidelities, color="b", s=10)
    ax1.set_title("Fidelity")
    ax1.set_title("a)", loc='left', fontsize='medium')
    ax1.set_xlabel("Instance")

    # ----> Scatter plot of infidelity per instance (log scale) <----
    ax2.scatter(range(len(infidelity)), infidelity, color="r", s=10)
    ax2.set_yscale("log")  # Log scale to emphasize smaller infidelities
    ax2.set_title("1 - Fidelity (log scale)")
    ax2.set_title("b)", loc='left', fontsize='medium')
    ax2.set_xlabel("Instance")

    # ----> Scatter plot of sum of rewards per instance <----
    ax3.scatter(range(len(rewards)), rewards, color="g", s=10)
    ax3.set_title("Sum of Rewards")
    ax3.set_title("c)", loc='left', fontsize='medium')
    ax3.set_xlabel("Instance")

    # Adjust subplot layout for better spacing
    plt.tight_layout()

    # Save or display the generated figure
    if save_dir:
        plt.savefig(os.path.join(save_dir, plot_filename))  # Ensure cross-platform path handling
        plt.close(fig)  # Close the figure to prevent redundant rendering
    else:
        plt.show()  # Display the figure interactively

def inference_distribution(save_dir, fidelities, bin_step, gate):
    # -------------------------------> Fidelity Histogram Plot <-------------------------------------

    # Define bin edges for histogram plotting with a step of 0.1
    bins = np.arange(0.0, 1.1, bin_step)  # Bins range from 0.0 to 1.0 in 0.1 increments

    # Set histogram title (if gate information is provided)
    count_title = f'[{gate}] Number of Occurrences vs Fidelity'

    # Compute histogram bin counts
    counts, bin_edges = np.histogram(fidelities, bins=bins)

    # Create a new figure for the histogram
    plt.figure(figsize=(8, 6))
    plt.bar(bin_edges[:-1], counts, width=0.1, align='edge', edgecolor='black')

    # Set labels and title for the histogram
    plt.xlabel("Fidelity")
    plt.ylabel("Number of Occurrences")
    plt.title(count_title)

    # Set x-axis ticks to match bin intervals
    plt.xticks(bins)

    # Enable grid lines on the y-axis for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to prevent overlapping labels
    plt.tight_layout()

    # Save or display the histogram
    if save_dir:
        plt.savefig(os.path.join(save_dir, "occurrences.png"))  # Use os.path.join() for proper file paths
    else:
        plt.show()  # Display the histogram interactively

def inference_bloch_sphere(save_dir, u_target_list, fidelities, gate):
    ## Plot Bloch Sphere for the following gates
    gates_to_plot = (Rx, Ry, RandomSU2)
    if isinstance(gate, gates_to_plot):
        # Initialize a dictionary to store quantum states categorized by fidelity bins (ranges)
        # Bins are defined in increments of 0.1 from 0.0 to 1.0
        fidelity_bins = {round(i, 1): [] for i in np.arange(0.0, 1.1, 0.1)}

        # Categorize unitary-transformed states based on their fidelity
        for unitary, fidelity in zip(u_target_list, fidelities):
            # Determine the fidelity bin (e.g., 0.2, 0.3, ..., 0.9)
            bin_range = np.floor(fidelity * 10) / 10  # Maps fidelity to its lower 0.1 range
            # Append the transformed quantum state (U|0⟩) to the corresponding bin
            fidelity_bins[bin_range].append(np.matmul(unitary, np.array([1, 0])))

        # Generate a Bloch sphere visualization for each fidelity bin
        for bin_range, states in fidelity_bins.items():
            if states:  # Only create a Bloch sphere if there are quantum states in the bin
                # Convert states to QuTiP quantum objects for visualization
                q_objs = [qutip.Qobj(state) for state in states]

                # Create a Bloch sphere object and add the quantum states
                bloch_sphere = qutip.Bloch()
                bloch_sphere.add_states(q_objs)

                # Save the visualization to a file if a save directory is specified
                if save_dir:
                    # Format filename by replacing decimal points in bin range with underscores
                    filename = f"Bloch_fidelity_in_bin_{str(bin_range).replace('.', '_')}"
                    bloch_sphere.save(name=os.path.join(save_dir, filename), format="png")
                else:
                    # Display the Bloch sphere interactively
                    bloch_sphere.show()

def multiple_inference_visuals(df, figure_title, save_dir, plot_filename, bin_step=0.1,
                           gate=None, perform_analysis=False):
    # Extract fidelity, rewards, and episode IDs from the DataFrame
    fidelities = np.array(df.iloc[:, 0])  # Fidelity values per episode
    rewards = np.array(df.iloc[:, 1])  # Rewards per episode
    episode_ids = np.array(df.iloc[:, 6])  # Episode indices
    u_target_list = df.iloc[:, 5]

    # Extract U_target matrix and action sequences from the dataset
    # Previous implementation involved applying preprocessing functions.
    # Here, we directly extract them from the DataFrame.
    if perform_analysis:
        actions_array = np.array(df.iloc[:, 2].tolist())
        action_stats_analysis(actions_array=actions_array,final_fidelity_per_episode=fidelities,save_dir=save_dir)


    # Compute infidelity as 1 - fidelity for each episode
    infidelity = np.array([1 - fidelities[i] for i in range(len(episode_ids))])

    # Round numerical values for better visualization
    rounding_precision = 6
    fidelities = np.round(fidelities, rounding_precision)
    infidelity = np.round(infidelity, rounding_precision)
    rewards = np.round(rewards, rounding_precision)

    inference_plot(fidelities, infidelity, rewards, figure_title, save_dir, plot_filename)
    inference_distribution(save_dir, fidelities, bin_step=bin_step, gate=gate)
    inference_bloch_sphere(save_dir, u_target_list, fidelities, gate)

# if __name__ == "__main__":
#     save_dir = RESULTS_DIR + "2024-02-27_19-31-17_H/"
#     plot_data(save_dir, episode_length=2, figure_title="")

