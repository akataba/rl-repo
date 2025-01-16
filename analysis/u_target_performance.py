from relaqs import RESULTS_DIR
from relaqs.api import load_pickled_env_data

path = '2025-01-16_11-27-21_arbitrary-gate'
data_path = RESULTS_DIR + path + '/env_data.pkl'

df = load_pickled_env_data(data_path)

fidelity = df["Fidelity"]
reward = df["Rewards"]
actions = df["Actions"]
operator = df["Operator"]
target_operator = df["Target Operator"]
episode_id = df["Episode Id"]

n_lowest = 10

lowest_fidelities = fidelity.nsmallest(n_lowest).values
lowest_fidelities_indices = fidelity.nsmallest(n_lowest).index
lowest_fidelity_target_operators = target_operator.iloc[lowest_fidelities_indices]

print("Target operators corresponding to the worst fidelity:")
for i in range(n_lowest):
    print("Fidelity:", lowest_fidelities[i])
    print(lowest_fidelity_target_operators.iloc[i], "\n")
