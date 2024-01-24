import sys
sys.path.append('./src/')

import ray
from ray.rllib.algorithms.ddpg import DDPGConfig
from ray.tune.registry import register_env
from relaqs.environments.gate_synth_env_rllib_Haar import TwoQubitGateSynth
from relaqs.save_results import SaveResults
from relaqs.plot_data import plot_data

from relaqs.quantum_noise_data.get_data import get_month_of_single_qubit_data, get_month_of_all_qubit_data
from relaqs import quantum_noise_data
from relaqs import QUANTUM_NOISE_DATA_DIR
from relaqs import RESULTS_DIR

from qutip.superoperator import liouvillian, spre, spost
from qutip import Qobj, tensor
from qutip.operators import *
from qutip import cnot, cphase

import numpy as np
import scipy.linalg as la
import random

import datetime


X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.array([[1, 0], [0, 1]])
Y = np.array([[0, 1j], [-1j, 0]])


#two-qubit single qubit gates
II = tensor(Qobj(I),Qobj(I)).data.toarray()
X1 = tensor(Qobj(X),Qobj(I)).data.toarray()
X2 = tensor(Qobj(I),Qobj(X)).data.toarray()
Y1 = tensor(Qobj(Y),Qobj(I)).data.toarray()
Y2 = tensor(Qobj(I),Qobj(Y)).data.toarray()
Z1 = tensor(Qobj(Z),Qobj(I)).data.toarray()
Z2 = tensor(Qobj(I),Qobj(Z)).data.toarray()

#two-qubit gate basis
XX = tensor(Qobj(X),Qobj(X)).data.toarray()
YY = tensor(Qobj(Y),Qobj(Y)).data.toarray()
ZZ = tensor(Qobj(Z),Qobj(Z)).data.toarray()

tx = random.random()-0.5
ty = random.random()-0.5
tz = random.random()-0.5
CanonicalOp = la.expm(np.pi/2*1j*(tx*XX+ty*YY+tz*ZZ))

def env_creator(config):
    return TwoQubitGateSynth(config)

# def save_grad_to_file(resultdict):
#     try:
#         policydict = resultdict["default_policy"]
#         stats = policydict["learner_stats"]
#         grad_gnorm = stats["grad_gnorm"]
#         with open("gradfile", "a") as f:
#             f.write(f"{grad_gnorm}\n")
#     except KeyError:
#         pass
        # print(f"Failed to extract grad_gnorm from: {resultdict}")

def inject_logging(alg, logging_func):
    og_ts = alg.training_step
    def new_training_step():
        result = og_ts()
        # do logging here
        logging_func(result)
        return result
    alg.training_step = new_training_step

def run(n_training_iterations=1, save=True, plot=True):
    ray.init(num_gpus=1)
    # ray.init()
    try:
        register_env("my_env", env_creator)


        # ---------------------> Configure algorithm and Environment <-------------------------
        alg_config = DDPGConfig().training().resources(num_gpus=1)
        # alg_config = DDPGConfig()
        alg_config.framework("torch")
        
        env_config = TwoQubitGateSynth.get_default_env_config()
        env_config["tx"] = tx
        env_config["ty"] = ty
        env_config["tz"] = tz
        env_config["U_target"] = CanonicalOp

        alg_config.environment("my_env", env_config=env_config)
    
        alg_config.rollouts(batch_mode="complete_episodes")
        alg_config.train_batch_size = TwoQubitGateSynth.get_default_env_config()["steps_per_Haar"]

        ### working 1-3 sets
        alg_config.actor_lr = 5e-5
        alg_config.critic_lr = 5e-5

        alg_config.actor_hidden_activation = "relu"
        alg_config.critic_hidden_activation = "relu"
        alg_config.num_steps_sampled_before_learning_starts = 600000
        # alg_config.actor_hiddens = [500,20000,500]
        # alg_config.critic_hiddens = [500,20000,500]
        alg_config.actor_hiddens = [1000, 100]
        alg_config.critic_hiddens = [1000, 100]
        # alg_config.exploration_config["scale_timesteps"] = 200000
        alg_config.exploration_config["scale_timesteps"] = 1200000
        print(alg_config.algo_class)
        print(alg_config["framework"])

        alg = alg_config.build()
        # inject_logging(alg, save_grad_to_file)
        # ---------------------------------------------------------------------
        list_of_results = []
        
        datetimeStr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S/")
        # ---------------------> Train Agent <-------------------------
        for ii in range(n_training_iterations):
            result = alg.train()
            list_of_results.append(result['hist_stats'])
            if np.mod(ii,5)==0:
                print("currently",ii,"/",n_training_iterations)
                      
                # ---------------------> Save Results <-------------------------
                if save is True:
                    env = alg.workers.local_worker().env
                    sr = SaveResults(env, alg, results=list_of_results, save_path = RESULTS_DIR + "two-qubit gates/"+"canonicalOp" + datetimeStr)
                    save_dir = sr.save_results()
                    print("Results saved to:", save_dir)
                # --------------------------------------------------------------

                # ---------------------> Plot Data <-------------------------
                if plot is True:
                    assert save is True, "If plot=True, then save must also be set to True"
                    plot_data(save_dir, episode_length=alg._episode_history[0].episode_length)
                    print("Plots Created")
                # --------------------------------------------------------------

        # -------------------------------------------------------------

    finally:
        ray.shutdown()

if __name__ == "__main__":
    n_training_iterations = 1280
    save = True
    plot = True
    run(n_training_iterations, save, plot)
