from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data, plot_results
import numpy as np
from relaqs.api.utils import do_inferencing, get_best_episode_information
from relaqs.api.gates import X, H
from relaqs.api.utils import (
    run_noisy_one_qubit_experiment,
    sample_noise_parameters,
    return_env_from_alg
)

best_fidelities_found = []
for _ in range(6):
    n_training_iterations = 250
    n_episodes_for_inferencing= 400
    figure_title ="Inferencing on multiple noisy environments with different detuning noise for X gate"
    noise_file = "april/ibmq_belem_month_is_4.json"
    noise_file_2 = "april/ibmq_quito_month_is_4.json"
    path_to_detuning = "qubit_detuning_data.json"

    # -----------------------> Training model <------------------------
    alg, list_of_results = run_noisy_one_qubit_experiment(X(), 
        n_training_iterations, 
        noise_file=noise_file
        )

    # ----------------------- Creating new environment with new detuning -------------------------------
    env = return_env_from_alg(alg)
    t1_list,t2_list,_  = sample_noise_parameters(noise_file_2, detuning_noise_file=path_to_detuning)
    detuning_list = np.random.normal(1e8, 1e12, 9).tolist()
    # t2_list = np.random.normal(1e-9, 1e-5, 135)x
    env.relaxation_rates_list = [np.reciprocal(t1_list).tolist(), np.reciprocal(t2_list).tolist()]
    env.delta = detuning_list 

    # -----------------------> Inferencing <---------------------------
    inferencing_env, inferencing_alg = do_inferencing(alg, n_episodes_for_inferencing,quantum_noise_file_path=noise_file_2)

    # -------------------> Save Inferencing Results <---------------------------------------
    sr = SaveResults(inferencing_env, inferencing_alg)
    save_dir = sr.save_results()
    print("Results saved to:", save_dir)
    # best_episode_information = get_best_episode_information(save_dir + "env_data.csv")
    best_episode_information = get_best_episode_information(save_dir + "env_data.pkl")

    print("Fidelities from best epsiode: ", [best_episode_information.iloc[0,0], best_episode_information.iloc[1,0]])
    best_fidelities_found.append((best_episode_information.iloc[0,0],best_episode_information.iloc[1,0] ))
    best_fidelity_tuple = str((best_episode_information.iloc[0,0],best_episode_information.iloc[1,0]))
    best_fidelity_file = "best_fidelities.txt"
    with open(save_dir + best_fidelity_file, 'w') as file:
        file.write(best_fidelity_tuple)


    # ---------------------> Plot Data <-------------------------------------------
    plot_data(save_dir, episode_length=inferencing_alg._episode_history[0].episode_length, figure_title=figure_title)
    

print(best_fidelities_found)


