import ray
import numpy as np
import pandas as pd
from numpy.linalg import eigvalsh
from scipy.linalg import sqrtm
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ddpg import DDPGConfig
from relaqs import RESULTS_DIR
from relaqs.quantum_noise_data.get_data import (get_month_of_all_qubit_data, get_single_qubit_detuning)
from relaqs.api.callbacks import GateSynthesisCallbacks
from relaqs import QUANTUM_NOISE_DATA_DIR
from qutip.operators import *
from datetime import datetime
import re
import qutip
import os
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

vec = lambda X : X.reshape(-1, 1, order="F") # vectorization operation, column-order. X is a numpy array.
vec_inverse = lambda X : X.reshape(int(np.sqrt(X.shape[0])),
                                   int(np.sqrt(X.shape[0])),
                                   order="F") # inverse vectorization operation, column-order. X is a numpy array.

def get_time():
    return datetime.now()

def preprocess_actions(matrix_str):
    """
    Preprocesses a matrix string by formatting spaces and commas correctly.

    This function is designed to clean up a given matrix string representation
    by ensuring:
    1. Multiple spaces are replaced with a single comma (for CSV-like formatting).
    2. Extraneous commas inside brackets (e.g., "[,0.4,...,]") are removed.

    Args:
        matrix_str (str): The input matrix string containing numerical values.

    Returns:
        str: The cleaned matrix string with proper formatting.
    """

    # Remove any leading and trailing whitespace characters from the input string
    matrix_str = matrix_str.strip()

    # Replace all occurrences of one or more whitespace characters (\s+) with a single comma
    # This ensures that values are properly separated by commas instead of spaces
    matrix_str = re.sub(r"\s+", ",", matrix_str)

    # Remove any extraneous commas immediately after an opening bracket or before a closing bracket
    # Example: "[,0.4,0.5,]" should be corrected to "[0.4,0.5]"
    matrix_str = matrix_str.replace("[,", "[").replace(",]", "]")

    return matrix_str


def preprocess_matrix_string(matrix_str):
    """
    Preprocesses a matrix string representation to ensure consistent formatting.

    This function is responsible for:
    1. Removing any newline characters ('\n') to keep the entire matrix on one line.
    2. Adding commas after complex numbers to maintain proper list formatting.
    3. Ensuring that adjacent bracketed lists (e.g., "] [") are correctly formatted.

    Args:
        matrix_str (str): The input string representing a matrix.

    Returns:
        str: The cleaned and properly formatted matrix string.
    """

    # Step 1: Remove any newline characters to ensure the matrix is represented in a single line
    matrix_str = matrix_str.replace('\n', '')

    # Step 2: Ensure that complex numbers (ending with 'j') are correctly followed by a comma
    # Example: "1+2j 3+4j" → "1+2j, 3+4j"
    matrix_str = matrix_str.replace('j ', 'j, ')

    # Step 3: Ensure that separate lists inside the matrix are properly delimited
    # Example: "] [1, 2, 3]" → "], [1, 2, 3]"
    matrix_str = matrix_str.replace('] [', '], [')

    return matrix_str

def check_unitary(matrices):
    """
    Check if a single matrix or all matrices in a list are unitary.

    Parameters:
    matrices (np.ndarray or list of np.ndarray): A single matrix or a list of matrices.

    Returns:
    bool: True if the matrix is unitary or if all matrices in the list are unitary, False otherwise.
    """

    def is_unitary(matrix):
        """Helper function to check if a single matrix is unitary."""
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a numpy array.")

        if matrix.shape[0] != matrix.shape[1]:
            return False  # Must be square

        identity_mat = np.eye(matrix.shape[0], dtype=np.complex128)
        return np.allclose(matrix.conj().T @ matrix, identity_mat) and np.allclose(matrix @ matrix.conj().T, identity_mat)

    # If input is a single matrix, return its unitary check result
    if isinstance(matrices, np.ndarray):
        return is_unitary(matrices)

    # If input is a list, return True only if all matrices are unitary
    elif isinstance(matrices, list):
        return all(is_unitary(matrix) for matrix in matrices)

    else:
        raise TypeError("Input must be a numpy array or a list of numpy arrays.")

def visualize_gates(gates, save_dir=None):
    """
    Visualizes a list of quantum gates on the Bloch sphere.

    This function takes a list of quantum gates (unitary matrices), applies them
    to the |0⟩ state, and plots the resulting quantum states on a Bloch sphere.

    Args:
        gates (list of np.ndarray): A list of 2x2 unitary matrices representing quantum gates.
        save_dir (str, optional): Directory to save the Bloch sphere visualization.
                                  If None, the visualization is displayed instead.

    Returns:
        None
    """

    # Initialize an empty list to store Bloch vectors
    bloch_vectors = []

    # Apply each gate to the |0⟩ state and compute the resulting quantum state
    for gate in gates:
        vector = np.matmul(gate, np.array([1, 0]))  # Apply gate to |0⟩ = [1, 0]^T
        q_obj = qutip.Qobj(vector)  # Convert the resulting vector to a QuTiP quantum object
        bloch_vectors.append(q_obj)  # Store the quantum object representation

    # Create a Bloch sphere object for visualization
    bloch_sphere = qutip.Bloch()

    # Add the computed quantum states to the Bloch sphere
    bloch_sphere.add_states(bloch_vectors)

    # Default Bloch sphere view
    bloch_sphere.view = [-60, 30]

    # Optional: Rotate Bloch Sphere 180 degrees to see it from the back
    # bloch_sphere.view = [120, 30]

    # Prevent cropping by modifying rendering settings
    bloch_sphere.frame_alpha = 0.3  # Make the Bloch sphere frame slightly transparent
    bloch_sphere.font_size = 10  # Reduce font size to prevent label overlap
    bloch_sphere.scale = [1.0]  # Maintain default scaling to avoid zooming issues

    # **Save or Show the Bloch Sphere Visualization**
    if save_dir:
        # Generate a filename for the saved image
        filename = "Visualisation_of_Gates.png"

        # Save the Bloch sphere visualization as a PNG file in the specified directory
        bloch_sphere.save(name=os.path.join(save_dir, filename), format="png")
    else:
        # Display the Bloch sphere visualization interactively
        bloch_sphere.show()


def get_last_episode_step(raw_data):
    # Group by 'Episode Id' and get the last index for each unique episode
    last_indices = raw_data.groupby('Episode Id').tail(1).index.tolist()
    return last_indices

def perform_action_analysis(df):
    # Get the last step of each episode in the dataset
    last_step = get_last_episode_step(df)

    # Extract only the data corresponding to the last step of each episode
    df_filtered = df.loc[last_step].reset_index(drop=True)

    # Preprocess the action strings to ensure proper formatting
    preProcessed_actions = df_filtered.iloc[:, 2].apply(preprocess_actions)

    # Convert the processed action strings into NumPy arrays
    # Each action string (which represents a list) is evaluated and converted into an array
    actions_array = [np.array(eval(m)) for m in preProcessed_actions]

    # Compute the index corresponding to the last 10% of the episode data
    last_10_percent_idx = int(0.9 * len(actions_array))

    # Extract the fidelity values corresponding to the last step of each episode
    fidelity_array = df_filtered.iloc[:, 0]

    # Extract the most recent 10% of fidelity and action data for analysis
    recent_fidelity = fidelity_array[last_10_percent_idx:]
    recent_actions = np.array(actions_array[last_10_percent_idx:])

    # Perform statistical analysis on the recent action-fidelity relationship
    action_stats_analysis(actions_array=recent_actions, final_fidelity_per_episode=recent_fidelity, save_dir=save_dir)


def action_stats_analysis(actions_array, final_fidelity_per_episode, save_dir):
    """
        Performs correlation and regression analysis on action parameters and fidelity.

        This function analyzes the relationship between different action parameters
        (gamma_magnitude, gamma_phase, alpha) and the final fidelity per episode.

        The analysis includes:
        1. **Pearson Correlation Analysis**:
            - Measures the strength and direction of the linear relationship between
              each action parameter and the fidelity.
            - Reports the correlation coefficient and statistical significance (p-value).
        2. **Linear Regression Analysis**:
            - Fits a linear model to predict fidelity based on the action parameters.
            - Extracts regression coefficients to quantify each parameter's influence.
        3. **Results Export**:
            - Saves the correlation results and regression coefficients to a text file
              inside the specified save directory.

        Args:
            actions_array (np.ndarray): A NumPy array of shape (n_episodes, 3),
                                        where each row represents an episode's
                                        (gamma_magnitude, gamma_phase, alpha).
            final_fidelity_per_episode (np.ndarray): A 1D NumPy array of shape (n_episodes,)
                                                     containing fidelity values per episode.
            save_dir (str): Directory where the analysis results will be saved.

        Returns:
            None (Results are saved as a text file in `save_dir`).
        """

    # Define column names for the action parameters
    action_cols = ["gamma_magnitude", "gamma_phase", "alpha"]

    # 1. Convert input data to a Pandas DataFrame for easier analysis
    data = pd.DataFrame({
        "gamma_magnitude": actions_array[:, 0],  # Extract first column (gamma magnitude)
        "gamma_phase": actions_array[:, 1],  # Extract second column (gamma phase)
        "alpha": actions_array[:, 2],  # Extract third column (alpha)
        "Fidelity": final_fidelity_per_episode  # Target variable (fidelity per episode)
    })

    # 2. Correlation Analysis (Computing Pearson correlation coefficient for each action parameter)
    correlations = {}  # Dictionary to store correlation results

    for col in action_cols:
        # Compute Pearson correlation and p-value between the action and fidelity
        corr, p_value = pearsonr(data[col], data["Fidelity"])
        correlations[col] = {"Correlation": corr, "P-Value": p_value}

    # 3. Linear Regression Analysis
    X = actions_array  # Independent variables (action parameters)
    y = final_fidelity_per_episode  # Dependent variable (fidelity)

    # Initialize and fit a linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Extract regression coefficients for each action parameter
    coefficients = model.coef_

    # 4. Save results to a text file
    with open(save_dir + "actions_analysis.txt", "w") as f:
        # Write correlation analysis results
        f.write("Correlation Analysis:\n")
        f.write(pd.DataFrame(correlations).T.to_string())  # Convert dictionary to readable table
        f.write("\n\nLinear Regression Coefficients:\n")

        # Write regression coefficients for each action parameter
        for i, action in enumerate(action_cols):
            f.write(f"{action}: {coefficients[i]:.4f}\n")

        # Write explanatory insights on statistical analysis
        f.write("\nInsights:\n")

        # 1. Pearson Correlation Coefficient Explanation
        f.write("1. Pearson Correlation Coefficient:\n")
        f.write("   - Measures the strength and direction of the relationship between actions and fidelity.\n")
        f.write(
            "   - Values close to +1 indicate a positive correlation; -1 indicates a negative correlation; 0 means no correlation.\n")
        f.write("   - A high absolute value suggests a strong impact of the action on fidelity.\n")

        # 2. P-Value Explanation
        f.write("\n2. P-Value (Statistical Significance):\n")
        f.write("   - Determines if the correlation is statistically significant or due to chance.\n")
        f.write(
            "   - A p-value < 0.05 indicates a meaningful relationship; >= 0.05 suggests it may be due to random variations.\n")
        f.write("   - A high correlation but large p-value means uncertainty in the relationship.\n\n")

        # 3. Linear Regression Coefficients Explanation
        f.write("\n3. Linear Regression Coefficients:\n")
        f.write("   - Represents the magnitude and direction of influence of actions on fidelity.\n")
        f.write("   - A larger absolute coefficient means greater impact on fidelity.\n")
        f.write(
            "   - Positive values indicate increasing the action increases fidelity, negative values indicate the opposite.\n")

        # 4. Interpretation of Combined Metrics
        f.write("\n4. Combined Interpretation:\n")
        f.write("   - High correlation and low p-value suggest a strong and reliable relationship.\n")
        f.write("   - High regression coefficients mean actions significantly affect fidelity.\n")
        f.write("   - Consider both correlation strength and statistical significance when making decisions.\n")

def network_config_creator(alg_config):

    network_config = {
        "actor_lr":  alg_config.actor_lr,
        "actor_hidden_activation": alg_config.actor_hidden_activation,
        "critic_hidden_activation": alg_config.critic_hidden_activation,
        "critc_lr": alg_config.critic_lr,
        "actor_num_hidden_layers": len(alg_config.actor_hiddens),
        "actor_num_hidden_neurons":  alg_config.actor_hiddens[0],
        "critc_num_hidden_layers": len(alg_config.critic_hiddens),
        "critc_num_hidden_neurons":  alg_config.critic_hiddens[0],
        "num_steps_sampled_before_learning_starts": alg_config.num_steps_sampled_before_learning_starts,
        "twin_q" : alg_config.twin_q,
        "train_batch_size": alg_config.train_batch_size,
    }

    return network_config

def config_table(env_config, alg_config, filepath, continue_training=False, original_training_date = None):
    filtered_env_config = {}
    filtered_explor_config = {}
    network_config = network_config_creator(alg_config)

    env_config_default = {
        "num_Haar_basis": 1,
        "steps_per_Haar": 2,
    }


    network_config_default = {
        "actor_lr": 1e-3,
        "actor_hidden_activation": "relu",
        "critic_hidden_activation": "relu",
        "critc_lr": 1e-3,
        "actor_num_hidden_layers": "2",
        "actor_num_hidden_neurons": "[400,300]",
        "critc_num_hidden_layers": "2",
        "critc_num_hidden_neurons": "[400,300]",
        "num_steps_sampled_before_learning_starts": 1500,
        "twin_q": False,
        "train_batch_size": 256,
    }

    explor_config_default = {
        "random_timesteps": 1000,
        "ou_base_scale": 0.1,
        "ou_theta": 0.15,
        "ou_sigma": 0.2,
        "initial_scale": 1.0,
        "scale_timesteps": 10000
    }

    for key in env_config_default.keys():
        filtered_env_config[key] = env_config[key]

    for key in explor_config_default.keys():
        filtered_explor_config[key] = alg_config.exploration_config[key]

    env_data = {
        "Config Name": list(filtered_env_config.keys()),
        "Current Value": list(filtered_env_config.values()),
        "Default Value": list(env_config_default.values()),
    }

    network_data = {
        "Config Name": list(network_config.keys()),
        "Current Value": list(network_config.values()),
        "Default Value": list(network_config_default.values()),
    }

    explor_data = {
        "Config Name": list(filtered_explor_config.keys()),
        "Current Value": list(filtered_explor_config.values()),
        "Default Value": list(explor_config_default.values()),
    }

    env_df = pd.DataFrame(env_data)
    network_df = pd.DataFrame(network_data)
    explor_df = pd.DataFrame(explor_data)

    with open(filepath + "ddpg_config_table.txt", "w") as f:
        # Write the table header with a border
        f.write("+------------------------------------------------+----------------------+--------------------+\n")
        f.write("|                  Config Name                   |     Current Value    |    Default Value   |\n")
        f.write("+------------------------------------------------+----------------------+--------------------+\n")

        for index, row in env_df.iterrows():
            f.write(f"| {row['Config Name']: <46} | {row['Current Value']: <21} | {row['Default Value']: <18} |\n")
        f.write("+------------------------------------------------+----------------------+--------------------+\n")

        for index, row in explor_df.iterrows():
            f.write(f"| {row['Config Name']: <46} | {row['Current Value']: <21} | {row['Default Value']: <18} |\n")
        f.write("+------------------------------------------------+----------------------+--------------------+\n")

        for index, row in network_df.iterrows():
            f.write(f"| {row['Config Name']: <46} | {row['Current Value']: <21} | {row['Default Value']: <18} |\n")
        f.write("+------------------------------------------------+----------------------+--------------------+\n")
        f.write(f"Continuation from previous training: {continue_training}\n")
        if continue_training:
            f.write(f"Training continued from results on: {original_training_date}\n")


def normalize(quantity, list_of_values):
    """ normalize quantity to [0, 1] range based on list of values """
    return (quantity - min(list_of_values) + 1E-15) / (max(list_of_values) - min(list_of_values) + 1E-15)

def polar_vec_to_complex_matrix(vec, return_flat=False):
    """ 
    The intended use of this function is to convert from the representation of the unitary
    in the agent's observation back to the unitary matrxi.

    Converts a vector of polar coordinates to a unitary matrix. 
    
    The vector is of the form: [r1, phi1, r2, phi2, ...]
    
    And the matrix is then: [-1 * r1 * exp(i * phi1 * 2pi),...] """
    # Convert polar coordinates to complex numbers
    complex_data = []
    for i in range(0, len(vec), 2):
        r = vec[i]
        phi = vec[i+1]
        z = -1 * r * np.exp(1j * phi * 2*np.pi) 
        complex_data.append(z)

    # Reshape into square matrix
    if not return_flat:
        matrix_dimension = int(np.sqrt(len(vec)))
        complex_data = np.array(complex_data).reshape((matrix_dimension, matrix_dimension))

    return complex_data

def superoperator_evolution(superop, dm):
    return vec_inverse(superop @ vec(dm))

def load_pickled_env_data(data_path):
    df = pd.read_pickle(data_path)
    return df

gate_fidelity = lambda U, V: float(np.abs(np.trace(U.conjugate().transpose() @ V))) / (U.shape[0])

def dm_fidelity(rho, sigma):
    assert np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).imag < 1e-8, f"Non-negligable imaginary component {np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).imag}"
    #return np.abs(np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))))**2
    return np.trace(sqrtm(sqrtm(rho) @ sigma @ sqrtm(rho))).real**2

def sample_noise_parameters(t1_t2_noise_file=None, detuning_noise_file=None, machine_name = None, qubit_label=None):
    # ---------------------> Get quantum noise data <-------------------------
    if t1_t2_noise_file is None:
        t1_list = np.random.uniform(40e-6, 200e-6, 100)
        t2_list = np.random.uniform(40e-6, 200e-6, 100)
    else:
        t1_list, t2_list = get_month_of_all_qubit_data(QUANTUM_NOISE_DATA_DIR + t1_t2_noise_file) # in seconds

    if detuning_noise_file is None:
        mean = 0
        std = 1e4
        sample_size = 100
        samples = np.random.normal(mean, std, sample_size)
        detunings = samples.tolist()
    else:
        detunings = get_single_qubit_detuning(QUANTUM_NOISE_DATA_DIR + detuning_noise_file, machine_name = machine_name, qubit_label=qubit_label)

    return list(t1_list), list(t2_list), detunings

def do_inferencing(alg, n_episodes_for_inferencing, quantum_noise_file_path):
    """
    alg: The trained model
    n_episodes_for_inferencing: Number of episodes to do during the training
    """
    
    assert n_episodes_for_inferencing > 0
    env = return_env_from_alg(alg)
    obs, info = env.reset()
    t1_list, t2_list, detuning_list = sample_noise_parameters(quantum_noise_file_path)
    env.relaxation_rates_list = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()]
    env.detuning_list = detuning_list
    num_episodes = 0
    episode_reward = 0.0
    print("Inferencing is starting ....")
    while num_episodes < n_episodes_for_inferencing:
        print("episode : ", num_episodes)
        # Compute an action (`a`).
        a = alg.compute_single_action(
            observation=obs,
            policy_id="default_policy",  # <- default value
        )
        # Send the computed action `a` to the env.
        obs, reward, done, truncated, _ = env.step(a)
        episode_reward += reward
        # Is the episode `done`? -> Reset.
        if done:
            print(f"Episode done: Total reward = {episode_reward}")
            obs, info = env.reset()
            num_episodes += 1
            episode_reward = 0.0
    return env, alg

def load_model(path):
    "path (str): Path to the file usually beginning with the word 'checkpoint' " 
    loaded_model = Algorithm.from_checkpoint(path)
    return loaded_model

def get_best_episode_information(filename):
    df = pd.read_csv(filename, names=['Fidelity', 'Reward', 'Actions', 'Flattened U', 'Episode Id'], header=0)
    fidelity = df.iloc[:, 0]
    max_fidelity_idx = fidelity.argmax()
    fidelity = df.iloc[max_fidelity_idx, 0]
    episode = df.iloc[max_fidelity_idx, 4]
    best_episode = df[df["Episode Id"] == episode]
    return best_episode

def get_best_actions(filename):
    best_episode = get_best_episode_information(filename)
    action_str_array = best_episode['Actions'].to_numpy()

    best_actions = []
    for actions_str in action_str_array:
        # Remove the brackets and split the string by spaces
        str_values = actions_str.strip('[]').split()

        # Convert the string values to float
        float_values = [float(value) for value in str_values]

        # Convert the list to a numpy array (optional)
        best_actions.append(float_values)
    return best_actions, best_episode['Fidelity'].to_numpy() 

def run(env_class, gate, n_training_iterations=1, noise_file=""):
    """Args
       gate (Gate type):
       n_training_iterations (int)
       noise_file (str):
    Returns
      alg (rllib.algorithms.algorithm)

    """
    ray.init()
    env_config = env_class.get_default_env_config()
    env_config["U_target"] = gate.get_matrix()

    # ---------------------> Get quantum noise data <-------------------------
    t1_list, t2_list, detuning_list = sample_noise_parameters(noise_file)

    env_config["relaxation_rates_list"] = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()] # using real T1 data
    env_config["detuning_list"] = detuning_list
    env_config["relaxation_ops"] = [sigmam(),sigmaz()]
    env_config["observation_space_size"] = 2*16 + 1 + 2 + 1 # 2*16 = (complex number)*(density matrix elements = 4)^2, + 1 for fidelity + 2 for relaxation rate + 1 for detuning
    env_config["verbose"] = True

    # ---------------------> Configure algorithm and Environment <-------------------------
    alg_config = DDPGConfig()
    alg_config.framework("torch")
    alg_config.environment(env_class, env_config=env_config)
    alg_config.rollouts(batch_mode="complete_episodes")
    alg_config.callbacks(GateSynthesisCallbacks)
    alg_config.train_batch_size = env_class.get_default_env_config()["steps_per_Haar"]

    ### working 1-3 sets
    alg_config.actor_lr = 4e-5
    alg_config.critic_lr = 5e-4

    alg_config.actor_hidden_activation = "relu"
    alg_config.critic_hidden_activation = "relu"
    alg_config.num_steps_sampled_before_learning_starts = 1000
    alg_config.actor_hiddens = [30,30,30]
    alg_config.exploration_config["scale_timesteps"] = 10000

    alg = alg_config.build()
    list_of_results = []
    for _ in range(n_training_iterations):
        result = alg.train()
        list_of_results.append(result['hist_stats'])

    ray.shutdown()

    return alg

def return_env_from_alg(alg):
    env = alg.workers.local_worker().env
    return env

def load_and_analyze_best_unitary(data_path, U_target):
    df = pd.read_csv(data_path, names=['Fidelity', 'Reward', 'Actions', 'Flattened U', 'Episode Id'], header=0)
    
    fidelity = df["Fidelity"]
    max_fidelity_idx = fidelity.argmax()
    best_flattened_unitary = eval(df.iloc[max_fidelity_idx, 3])

    best_fidelity_unitary = np.array([complex(x) for x in best_flattened_unitary]).reshape(4, 4)
    max_fidelity = fidelity.iloc[max_fidelity_idx]

    print("Max fidelity:", max_fidelity)
    print("Max unitary:", best_fidelity_unitary)

    zero = np.array([1, 0]).reshape(-1, 1)
    zero_dm = zero @ zero.T.conjugate()
    zero_dm_flat = zero_dm.reshape(-1, 1)

    dm = best_fidelity_unitary @ zero_dm_flat
    dm = dm.reshape(2, 2)
    print("Density Matrix:\n", dm)

    # Check trace = 1
    dm_diagonal = np.diagonal(dm)
    print("diagonal:", dm_diagonal)
    trace = sum(np.diagonal(dm))
    print("trace:", trace)

    # # Check that all eigenvalues are positive
    eigenvalues = eigvalsh(dm)
    print("eigenvalues:", eigenvalues)
    #assert (0 <= eigenvalues).all()

    psi = U_target.get_matrix() @ zero
    true_dm = psi @ psi.T.conjugate()
    print("true dm\n:", true_dm)

    print("Density matrix fidelity:", dm_fidelity(true_dm, dm))
