import numpy as np
from relaqs.environments import SingleQubitEnv, NoisySingleQubitEnv
from relaqs.api import gates

def actions_to_unitary(env, actions):
    env.step(actions)
    return env.U

# def fidelity_error_matrix(U1, U2):
#     #return np.abs(U1 - U2)
#     #return np.abs(np.abs(U1) - np.abs(U2))
#    return np.abs(U1) - np.abs(U2)

def fidelity_error_matrix(U_target, U):
    assert U_target.shape == U.shape
    assert U_target.shape[0] == U_target.shape[1]
    d = U_target.shape[0]
    error_matrix = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            #error = U_target[i, j].conj() * U_target[i, j] - U_target[i, j].conj() * U[i, j]
            #error = np.sqrt(U_target[i, j].conj() * U_target[i, j] - U_target[i, j].conj() * U[i, j])

            #error = U_target[i, j].conj() * U_target[i, j] - np.abs(U_target[i, j].conj() * U[i, j])

            #error = U_target[i, j].conj() * U_target[i, j] - U[i, j].conj() * U[i, j]
            #error = 1/2* np.sqrt(U_target[i, j].conj() * U_target[i, j]) - np.sqrt(U[i, j].conj() * U[i, j])

            #error = (U_target[i, j] - U[i, j]).conj().T * (U_target[i, j] - U[i, j])

            #error = np.abs(np.abs(U_target[i, j].conj() * U_target[i, j]) - np.abs(U[i, j].conj() * U[i, j])) # reverse triangle inequality
            #error = np.abs(np.abs(U_target[i, j].conj() * U_target[i, j]) - np.abs(U_target[i, j].conj() * U[i, j])) # reverse triangle inequality

            error = np.abs(U_target[i, j])**2 - np.abs(U[i, j])**2 # absolute difference
            #error_matrix[i, j] = np.abs(error)
            error_matrix[i, j] = error

    return (error_matrix / d)

if __name__ == "__main__":
    # Define env
    env = SingleQubitEnv(SingleQubitEnv.get_default_env_config())

    # Define actions
    gamma_magnitude = np.random.uniform(-1, 1)
    gamma_phase = np.random.uniform(-1, 1)
    alpha = np.random.uniform(-1, 1) 
    actions = [gamma_magnitude, gamma_phase, alpha]

    U = actions_to_unitary(env, actions)

    #env.U = gates.H().get_matrix()

    print("U\n", env.U)

    F = env.compute_fidelity()
    print("Fidelity: ", F)

    # Define target unitary
    U_target = env.U_target
    print("U_target\n", U_target)
    
    FEM = fidelity_error_matrix(env.U_target, env.U)
    print("FEM\n", FEM)
    print("sum abs(FEM)", np.sum(FEM))
    print("1 - F =", 1 - F)
    print("Is the math right:", np.sum(FEM) == 1 - F)
