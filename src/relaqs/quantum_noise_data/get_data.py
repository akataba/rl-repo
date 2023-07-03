import json
from random import randrange


def get_single_qubit_data(file_name, day=None, qubit_label=None):
    """ If day is None, returns T1 and T2 times for a random day.
    If qubit_label is None, returns T1 and T2 times for a random qubit. """
    noise_data = None
    with open(file_name) as user_file:
        noise_data = json.load(user_file)
    
    if day is None:
        day = str(randrange(27))

    if qubit_label is None:
        n_qubits = len(noise_data['T1'][day])
        qubit_label= randrange(n_qubits)
    
    t1 = noise_data['T1'][day][qubit_label]
    t2 = noise_data['T2'][day][qubit_label]

    return (t1, t2)

if __name__ == "__main__":
    t1, t2 = get_single_qubit_data('april/ibm_lagos_month_is_4.json')
