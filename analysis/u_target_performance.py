from relaqs import RESULTS_DIR
from relaqs.api import load_pickled_env_data
path = '2025-01-21_10-40-03_arbitrary-gate'
data_path = RESULTS_DIR + path + '/env_data.pkl'

df = load_pickled_env_data(data_path)

last_n = 1000

fidelity = df["Fidelity"].tail(last_n)
reward = df["Rewards"].tail(last_n)
actions = df["Actions"].tail(last_n)
operator = df["Operator"].tail(last_n)
target_operator = df["Target Operator"].tail(last_n)
episode_id = df["Episode Id"].tail(last_n)

n_lowest = 10
fidelities_np = fidelity.to_numpy()
lowest_fidelities = np.partition(fidelities_np, n_lowest)[:n_lowest]
lowest_fidelities_indices = np.argsort(fidelities_np)[:n_lowest]
lowest_fidelity_target_operators = target_operator.to_numpy()[lowest_fidelities_indices]

print("Target operators corresponding to the worst fidelity:")
for i in range(n_lowest):
    print("Fidelity:", lowest_fidelities[i])
    print(lowest_fidelity_target_operators[i], "\n")