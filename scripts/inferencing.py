from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data
import numpy as np
from relaqs.api.utils import do_inferencing, run
from relaqs.api.gates import H

noise_file = "april/ibmq_belem_month_is_4.json"
inferencing_noise_file = "april/ibmq_manila_month_is_4.json"
n_episodes_for_inferencing = 10
save = True
plot = True
figure_title = "Inferencing with model"
n_training_iterations = 1

# -----------------------> Training model <------------------------
alg = run(gate=H(), 
        n_training_iterations=n_training_iterations, 
        noise_file=noise_file
    )

# -----------------------> Inferencing <---------------------------
env, alg = do_inferencing(alg, n_episodes_for_inferencing,quantum_noise_file_path=inferencing_noise_file)

# -------------------> Save Inferencing Results <---------------------------------------
sr = SaveResults(env, alg)
save_dir = sr.save_results()
print("Results saved to:", save_dir)

# ---------------------> Plot Data <-------------------------------------------
assert save is True, "If plot=True, then save must also be set to True"

plot_data(save_dir, episode_length=alg._episode_history[0].episode_length, figure_title=figure_title)
print("Plots Created")
# --------------------------------------------------------------

