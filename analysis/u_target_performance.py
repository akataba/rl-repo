from relaqs import RESULTS_DIR
from relaqs.api import load_pickled_env_data
path = '2025-01-21_10-40-03_arbitrary-gate'
data_path = RESULTS_DIR + path + '/env_data.pkl'

df = load_pickled_env_data(data_path)

last_n = 1000

fidelity = df["Fidelity"][:last_n]
reward = df["Rewards"][:last_n]
actions = df["Actions"][:last_n]
operator = df["Operator"][:last_n]
target_operator = df["Target Operator"][:last_n]
episode_id = df["Episode Id"][:last_n]

n_lowest = 10

lowest_fidelities = fidelity.nsmallest(n_lowest).values
lowest_fidelities_indices = fidelity.nsmallest(n_lowest).index
lowest_fidelity_target_operators = target_operator.iloc[lowest_fidelities_indices]

print("Target operators corresponding to the worst fidelity:")
for i in range(n_lowest):
    print("Fidelity:", lowest_fidelities[i])
    print(lowest_fidelity_target_operators.iloc[i], "\n")
