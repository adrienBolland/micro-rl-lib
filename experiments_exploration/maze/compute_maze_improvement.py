import os
import time

import torch
import numpy as np
from functools import partial

from experiments_exploration.maze.initialize_problem import (initialize_experiment, compute_cumulative_reward_maze,
                                                             get_shaped_return)
from experiments_exploration.utils_multiprocessing import parallel_execution
from experiments_exploration.utils_pg import get_reinforce_grad, get_probability_stats


MC_SA = 2**19
NB_PROCESSES = 1

RETURN_RANGE = {
    "min_x": 0,
    "max_x": 1.,
    "pt_x": 100
}


NB_REINFORCE_SA = 2**3
REINFORCE_LR = 0.1


def experiment_probability_improvement(theta, oracle_return_list):
    # initialize
    env, model, agent, sampler = initialize_experiment()
    model.set_theta(theta)

    # do simulations
    state_batch, action_batch, reward_batch = sampler.sample(MC_SA)

    # compute cumulative reward
    cum_reward = compute_cumulative_reward_maze(env, agent, sampler, state_batch, action_batch, reward_batch)

    # compute the shaped cumulative rewards
    cum_reward_shaping = get_shaped_return(cum_reward)

    # extend the cumulative reward with the cumulative reward batch
    cum_reward = torch.cat([cum_reward, cum_reward_shaping], dim=-1)

    # compute policy score
    score = torch.sum(action_batch / theta - (1. - action_batch) / (1 - theta), dim=1)  # shape = (-1, 1)

    # compute the reinforce gradients
    max_norm = 1.
    reinforce_batch = get_reinforce_grad(score, cum_reward, NB_REINFORCE_SA, max_norm, baseline=True)

    # compute the probability of improvement of the return and the
    proba_improvement, avg_improvement, std_improvement = get_probability_stats([theta], reinforce_batch,
                                                                                REINFORCE_LR, oracle_return_list)

    return proba_improvement.tolist(), avg_improvement.tolist(), std_improvement.tolist()


def get_oracle_return(folder="plots_exploration"):
    with open(os.path.dirname(os.path.realpath(__file__)) + "/" + folder + "/pol_return_100_hist_002.npy", "rb") as f:
        array_return = torch.tensor(np.load(f))
        array_theta = torch.tensor(np.load(f)).reshape(1, -1)

        # extend the array of return to all reward shaping return
        array_return = torch.cat([array_return, get_shaped_return(array_return)], dim=-1)

    return [partial(evaluate_function, return_idx=idx, array_return=array_return, array_theta=array_theta)
            for idx in range(array_return.shape[-1])]


def evaluate_function(theta, return_idx, array_return, array_theta):
    array_theta.clamp_(min=0., max=1.)

    # Find the indices in the linspace corresponding to the input pair
    theta_idx = torch.argmin(torch.abs(array_theta - theta), dim=1, keepdim=True)

    # Evaluate the function at the found indices
    result = array_return[theta_idx, return_idx]

    return result


def pickle_policy_improvement(folder="plots_exploration"):
    array_theta = np.linspace(*RETURN_RANGE.values())

    parameters_enumeration = [[id_theta, (theta,)] for id_theta, theta in enumerate(array_theta)]
    solution_enum = parallel_execution(parameters_enumeration,
                                       partial(experiment_probability_improvement,
                                               oracle_return_list=get_oracle_return()),
                                       NB_PROCESSES)

    # fill the array with solutions
    improvement_probability = np.empty((len(array_theta)), dtype=list)
    average_improvement = np.empty((len(array_theta)), dtype=list)
    std_improvement = np.empty((len(array_theta)), dtype=list)
    for i_theta, (improvement_p_list, improvement_avg_list, improvement_std_list) in solution_enum:
        improvement_probability[i_theta] = improvement_p_list
        average_improvement[i_theta] = improvement_avg_list
        std_improvement[i_theta] = improvement_std_list

    # convert arrays of lists to full numpy arrays
    improvement_probability = np.array(improvement_probability.tolist())
    average_improvement = np.array(average_improvement.tolist())
    std_improvement = np.array(std_improvement.tolist())

    # save
    with open(os.path.dirname(os.path.realpath(__file__)) + "/" + folder + "/reinforce_with_baseline_100_hist_002.npy", "wb") as f:
        np.save(f, improvement_probability)
        np.save(f, average_improvement)
        np.save(f, std_improvement)
        np.save(f, array_theta)


if __name__ == "__main__":
    seed_value = 1234
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    nb_threads = 1  # nb of thread per process
    torch.set_num_threads(nb_threads)

    t = time.time()

    pickle_policy_improvement()

    print('Total computation time : ', time.time() - t)
