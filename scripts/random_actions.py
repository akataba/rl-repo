import csv
import numpy as np
from relaqs.environments import SingleQubitEnv, NoisySingleQubitEnv
from relaqs import RESULTS_DIR
from relaqs.api.utils import sample_noise_parameters
from qutip.operators import sigmam, sigmaz
import relaqs.api.gates as gates

def take_random_actions(env, n_episodes, save_file_name):
    """
    Takes random actions and saves the actions and fidleities to a csv file
    """
    action_history = []
    for _ in range(n_episodes):
        terminated = False
        while terminated is False:
            actions = env.action_space.sample()
            action_history.append(actions)
            state, reward, terminated, truncated, info = env.step(actions)
        env.reset()
    
    fidelities = [transition[0] for transition in env.transition_history]
    action_fidelity_data = [np.append(a, f) for a, f in zip(action_history, fidelities)]

    with open(RESULTS_DIR + "random_actions_fidelity/" + save_file_name, "w") as f:
        writer = csv.writer(f)
        for data in action_fidelity_data:
            writer.writerow(data)

if __name__ == "__main__":
    t1_list, t2_list, detuning_list = sample_noise_parameters()
    env_config = NoisySingleQubitEnv.get_default_env_config()
    env_config["detuning_list"] = detuning_list  # qubit detuning
    env_config["relaxation_rates_list"] =[t1_list, t2_list] # relaxation lists of list of floats to be sampled from when resetting environment. (10 usec)
    env_config["relaxation_ops"] = [sigmam(), sigmaz()]
    env_config["U_target"] = gates.Z().get_matrix()
    env = NoisySingleQubitEnv(env_config)
    n_episodes = 10000
    take_random_actions(env, n_episodes, save_file_name="changing_noise_Z.csv")
