import json
from random import randrange

def get_single_qubit_data(file_name, day=None, qubit_label=None):
    """ If day is None, returns T1 and T2 times for a random day.
    If qubit_label is None, returns T1 and T2 times for a random qubit. """
    noise_data = None
    with open(file_name) as f:
        noise_data = json.load(f)
    
    if day is None:
        day = str(randrange(start=1, stop=28))

    if qubit_label is None:
        n_qubits = len(noise_data['T1'][day])
        qubit_label= randrange(n_qubits)
    
    t1 = noise_data['T1'][day][qubit_label]
    t2 = noise_data['T2'][day][qubit_label]

    return (t1, t2)

def get_month_of_single_qubit_data(file_name, qubit_label):
    noise_data = None
    with open(file_name) as f:
        noise_data = json.load(f)
    days = list(range(1, 28))
    t1_list = [noise_data['T1'][str(day)][qubit_label] for day in days]
    t2_list = [noise_data['T2'][str(day)][qubit_label] for day in days]
    return t1_list, t2_list


if __name__ == "__main__":
    #t1, t2 = get_single_qubit_data('april/ibm_lagos_month_is_4.json')
    from relaqs import QUANTUM_NOISE_DATA_DIR
    path_to_file = QUANTUM_NOISE_DATA_DIR + "april/ibm_belem_month_is_4.json"
    t1_list, t2_list = get_month_of_single_qubit_data(path_to_file, qubit_label=0)
    print(t1_list)
    print(t2_list)
