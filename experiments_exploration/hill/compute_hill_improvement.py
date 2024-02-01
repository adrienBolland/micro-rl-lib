import os
import time

import torch
import numpy as np
from itertools import product
from functools import partial

from experiments_exploration.hill.initialize_problem import (initialize_experiment, compute_cumulative_reward_hill,
                                                             get_shaped_return)
from experiments_exploration.utils_multiprocessing import parallel_execution
from experiments_exploration.utils_pg import get_reinforce_grad, get_probability_stats

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


NB_REINFORCE_SA = 2**3
REINFORCE_LR = 0.1


def experiment_probability_improvement(k, std, oracle_return_list):
    env, det_agent_model, det_agent, stocha_agent_model, stocha_agent, sampler = initialize_experiment()
    det_agent_model.set_k(k)
    stocha_agent_model.set_sigma(std)

    # do simulations
    state_batch, action_batch, reward_batch = sampler.sample(MC_SA)

    # compute cumulative reward (for each intrinsic reward)
    cum_reward = compute_cumulative_reward_hill(env, stocha_agent, sampler, state_batch, action_batch, reward_batch)

    # compute the shaped cumulative rewards
    cum_reward_shaping = get_shaped_return(cum_reward)

    # extend the cumulative reward with the cumulative reward batch
    cum_reward = torch.cat([cum_reward, cum_reward_shaping], dim=-1)

    # compute policy score
    mu = k * (state_batch[:, :-1, :] - env.unwrapped._target)
    std += 1e-6
    score_mu = (action_batch - mu) / std**2
    score_k = score_mu * (state_batch[:, :-1, :] - env.unwrapped._target)
    score_std = (action_batch - mu)**2 / std**3 - 1 / std
    score = torch.sum(torch.stack([score_k, score_std], dim=-1), dim=1).squeeze()  # shape = (-1, 2)

    # compute the reinforce gradients
    max_norm = 1.
    reinforce_batch = get_reinforce_grad(score, cum_reward, NB_REINFORCE_SA, max_norm)

    # compute the probability of improvement of the return and the
    proba_improvement, avg_improvement, std_improvement = get_probability_stats([k, std], reinforce_batch,
                                                                                REINFORCE_LR, oracle_return_list)

    return proba_improvement.tolist(), avg_improvement.tolist(), std_improvement.tolist()


def get_oracle_return(folder="plots_exploration"):
    with open(os.path.dirname(os.path.realpath(__file__)) + "/" + folder + "/pol_return_200_hist.npy", "rb") as f:
        array_return = torch.tensor(np.load(f))
        array_k = torch.tensor(np.load(f)).reshape(1, -1)
        array_std = torch.tensor(np.load(f)).reshape(1, -1)

        # extend the array of return to all reward shaping return
        array_return = torch.cat([array_return, get_shaped_return(array_return)], dim=-1)

    return [partial(evaluate_function, return_idx=idx, array_return=array_return, array_k=array_k, array_std=array_std)
            for idx in range(array_return.shape[-1])]


def evaluate_function(position, return_idx, array_return, array_k, array_std):
    k, std = position.split(1, dim=-1)
    std.clamp_(min=0.)

    # Find the indices in the linspace corresponding to the input pair
    k_idx = torch.argmin(torch.abs(array_k - k), dim=1, keepdim=True)
    std_idx = torch.argmin(torch.abs(array_std - std), dim=1, keepdim=True)

    # Evaluate the function at the found indices
    result = array_return[std_idx, k_idx, return_idx]

    return result


def pickle_policy_improvement(folder="plots_exploration"):
    # generate the return values
    array_k = np.linspace(*RETURN_RANGE.values())
    array_std = np.linspace(*STD_RANGE.values())

    parameters_enumeration = [[(id_k, id_std), (k, std)]
                              for (id_k, k), (id_std, std) in product(enumerate(array_k), enumerate(array_std))]

    solution_enum = parallel_execution(parameters_enumeration,
                                       partial(experiment_probability_improvement,
                                               oracle_return_list=get_oracle_return()),
                                       NB_PROCESSES)

    # fill the array with solutions
    improvement_probability = np.empty((len(array_std), len(array_k)), dtype=list)
    average_improvement = np.empty((len(array_std), len(array_k)), dtype=list)
    std_improvement = np.empty((len(array_std), len(array_k)), dtype=list)
    for (i_k, i_std), (improvement_p_list, improvement_avg_list, improvement_std_list) in solution_enum:
        improvement_probability[i_std, i_k] = improvement_p_list
        average_improvement[i_std, i_k] = improvement_avg_list
        std_improvement[i_std, i_k] = improvement_std_list

    # convert arrays of lists to full numpy arrays
    improvement_probability = np.array(improvement_probability.tolist())
    average_improvement = np.array(average_improvement.tolist())

    # save
    with open(os.path.dirname(os.path.realpath(__file__)) + "/" + folder + "/reinforce_200_hist.npy", "wb") as f:
        np.save(f, improvement_probability)
        np.save(f, average_improvement)
        # np.save(f, std_improvement)
        np.save(f, array_k)
        np.save(f, array_std)


if __name__ == "__main__":
    seed_value = 1234
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)

    nb_threads = 1  # nb of thread per process
    torch.set_num_threads(nb_threads)

    t = time.time()

    pickle_policy_improvement()

    print('Total computation time : ', time.time() - t)
