import os
import time

import torch
import numpy as np
from itertools import product

from experiments_exploration.hill.initialize_problem import initialize_experiment, compute_cumulative_reward_hill
from experiments_exploration.utils_multiprocessing import parallel_execution


MC_SA = 2**12
NB_PROCESSES = 10

RETURN_RANGE = {
    "min_x": -4.,
    "max_x": 3.,
    "pt_x": 200
}

STD_RANGE = {
    "min_std": 0.,
    "max_std": 4.,
    "pt_std": 200
}


def experiment_return_hill(k, std):
    # initialize
    env, det_agent_model, det_agent, stocha_agent_model, stocha_agent, sampler = initialize_experiment()
    det_agent_model.set_k(k)
    stocha_agent_model.set_sigma(std)

    # do simulations
    state_batch, action_batch, reward_batch = sampler.sample(MC_SA)

    # compute cumulative reward
    cum_reward = compute_cumulative_reward_hill(env, stocha_agent, sampler, state_batch, action_batch, reward_batch)

    return torch.mean(cum_reward, dim=0)  # dim = (4,)


def pickle_policy_return(folder="plots_exploration"):
    # generate the return values
    array_k = np.linspace(*RETURN_RANGE.values())
    array_std = np.linspace(*STD_RANGE.values())

    parameters_enumeration = [[(id_k, id_std), (k, std)]
                              for (id_k, k), (id_std, std) in product(enumerate(array_k), enumerate(array_std))]
    solution_enum = parallel_execution(parameters_enumeration, experiment_return_hill, NB_PROCESSES)

    # fill the array with solutions
    array_return = np.empty((len(array_std), len(array_k), 4))
    for (i_k, i_std), return_np in solution_enum:
        array_return[i_std, i_k, :] = return_np

    # save
    with open(os.path.dirname(os.path.realpath(__file__)) + "/" + folder + "/pol_return_200_hist.npy", "wb") as f:
        np.save(f, array_return)
        np.save(f, array_k)
        np.save(f, array_std)


if __name__ == "__main__":
    seed_value = 1234
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    nb_threads = 1  # nb of thread per process
    torch.set_num_threads(nb_threads)

    t = time.time()

    pickle_policy_return()

    print('Total computation time : ', time.time() - t)
