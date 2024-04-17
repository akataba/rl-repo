"""
Example script to load pickled env data
"""

from relaqs import RESULTS_DIR
from relaqs.api import load_pickled_env_data

data_path = RESULTS_DIR + '2024-01-24_11-37-15_X/env_data.pkl'

df = load_pickled_env_data(data_path)

fidelity = df["Fidelity"]
reward = df["Rewards"]
actions = df["Actions"]
unitary = df["Operator"]
episode_id = df["Episode Id"]

max_fidelity_idx = fidelity.argmax()
max_fidelity = fidelity.iloc[max_fidelity_idx]

print("Max fidelity:", max_fidelity)