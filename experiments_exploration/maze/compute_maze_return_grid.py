import os
import time

import torch
import numpy as np

from experiments_exploration.maze.initialize_problem import initialize_experiment, compute_cumulative_reward_maze
from experiments_exploration.utils_multiprocessing import parallel_execution


MC_SA = 2**19
NB_PROCESSES = 1


RETURN_RANGE = {
    "min_x": 0,
    "max_x": 1.,
    "pt_x": 100
}


def experiment_return_maze(theta):
    # initialize
    env, model, agent, sampler = initialize_experiment()
    model.set_theta(theta)

    # do simulations
    state_batch, action_batch, reward_batch = sampler.sample(MC_SA)

    # compute cumulative reward
    cum_reward = compute_cumulative_reward_maze(env, agent, sampler, state_batch, action_batch, reward_batch)

    return torch.mean(cum_reward, dim=0)  # dim = (4,)


def pickle_policy_return(folder="plots_exploration"):
    # generate the return values
    array_theta = np.linspace(*RETURN_RANGE.values())

    parameters_enumeration = [[id_theta, (theta,)] for id_theta, theta in enumerate(array_theta)]
    solution_enum = parallel_execution(parameters_enumeration, experiment_return_maze, NB_PROCESSES)

    # fill the array with solutions
    array_return = np.empty((len(array_theta), 4))
    for i_theta, return_np in solution_enum:
        array_return[i_theta, :] = return_np

    # save
    with open(os.path.dirname(os.path.realpath(__file__)) + "/" + folder + "/pol_return_100_hist_002.npy", "wb") as f:
        np.save(f, array_return)
        np.save(f, array_theta)


if __name__ == "__main__":
    seed_value = 1234
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    nb_threads = 1  # nb of thread per process
    torch.set_num_threads(nb_threads)

    t = time.time()

    pickle_policy_return()

    print('Total computation time : ', time.time() - t)
