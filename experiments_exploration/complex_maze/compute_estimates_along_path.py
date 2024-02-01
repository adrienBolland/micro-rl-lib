import argparse
import os
import torch
import numpy as np

from copy import deepcopy

from algo.REINFORCE.REINFORCE import Reinforce
from experiments_exploration.complex_maze.initialize_problem import initialize_experiment
from exploration.IntrinsicReward.PolicyEntropy import PolicyEntropy
from exploration.IntrinsicReward.StateEntropy import StateEntropyGMM
from logger.DictLogger import DictLogger
from logger.TensorboardLogger import TensorboardLogger


def reinforce_experiment(exp_number, nb_reinforce_interation, learning_rate, batch_size, learning_int_reward_weight,
                         learning_entropy_weight, exp_nb_fit_estimation, exp_nb_mc_estimation, exp_int_reward_weight,
                         exp_entropy_weight, exp_period, compute_no_statistics, seed, verbose):
    # improvement threshold for the computation of P(improvement)
    improvement_threshold = 0.2

    # set seeds
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # set number of threads
    nb_threads = 1  # nb of thread per process
    torch.set_num_threads(nb_threads)

    # initialize the experiment
    _, env_ohe, model, agent, sampler = initialize_experiment(exp_number)

    # reward bonuses
    if exp_number == 1:
        intrinsic_reward_entropy = PolicyEntropy(env_ohe, agent).initialize()
    elif exp_number == 2 or exp_number == 3:
        intrinsic_reward_entropy = StateEntropyGMM(env_ohe, agent).initialize(min_clamp=-250, max_clamp=250)
    else:
        raise NotImplementedError

    # create a tensorboard log
    log_path = (os.path.dirname(os.path.realpath(__file__))
                + f"/logs/reinforce_lr_{learning_rate}_weight_{learning_entropy_weight}_int_{learning_int_reward_weight}")
    board_logger = TensorboardLogger(None, log_path)

    # create a dict logger
    dict_logger = DictLogger(["return", "learning-obj", "loss-reinforce", "loss-entropy", "grad-norm",
                              "optim-state-dict", "trajectory", "intrinsic-rewards"])

    # run the reinforce algo
    algo = Reinforce(env_ohe, sampler, {"policy": agent}, logger=dict_logger)
    algo.initialize(optimizer_parameters={'lr': learning_rate}, batch_size=batch_size)

    for it in range(nb_reinforce_interation):
        if verbose:
            print(f"---{it}---")

        # perform a reinforce iteration
        algo.set_reward_shaping([intrinsic_reward_entropy], [learning_int_reward_weight])
        algo.set_entropy_weight(learning_entropy_weight)
        algo.fit(1)

        # compute the statistics when it is a multiple of exp_period
        if it % exp_period:
            continue

        # get the reinforce log
        log_result = deepcopy(dict_logger.get_log())

        # get the parameters from the algo and model
        model_parameters = deepcopy(model.state_dict())
        optimizer_state = log_result.pop("optim-state-dict")

        # get the reward trajectories
        _, _, reward_batch = log_result.pop("trajectory")
        list_int_reward_batch = log_result.pop("intrinsic-rewards")  # same reward shaping, only the weights changed

        # set the experiment entropy weight
        algo.set_entropy_weight(exp_entropy_weight)

        # compute the return of each new policy when using reward shaping
        for (list_int_shaping, list_int_weight), name in zip([[[intrinsic_reward_entropy], [exp_int_reward_weight]],
                                                              [[intrinsic_reward_entropy], [0.]]],
                                                             ["grad-Hpi", "grad-J"]):
            if compute_no_statistics:
                break

            # perform and update
            nb_improvement_return = 0.
            nb_improvement_learning = 0.
            total_improvement_return = 0.
            total_improvement_learning = 0.

            # compute the values of the learning objective and return
            int_reward = torch.sum(torch.cat([r * w for r, w in zip(list_int_reward_batch, list_int_weight)], dim=-1),
                                   dim=-1, keepdim=True)

            return_theta = sampler.cumulative_reward(reward_batch)
            learning_theta = sampler.cumulative_reward(reward_batch + int_reward)

            # set the reward selection in reinforce
            algo.set_reward_shaping(list_int_shaping, list_int_weight)

            for mc_sim in range(exp_nb_mc_estimation):
                # compute statistics from network
                algo.fit(exp_nb_fit_estimation)

                # evaluate the new value of the return and learning objective
                log_result_ = dict_logger.get_log()

                # compute the statistics over the improvement
                delta_return = log_result_["return"] - return_theta
                delta_learning = log_result_["learning-obj"] - learning_theta

                if delta_return > improvement_threshold:
                    nb_improvement_return += 1.
                if delta_learning > improvement_threshold:
                    nb_improvement_learning += 1.

                total_improvement_return += delta_return
                total_improvement_learning += delta_learning

                # reset the algo and model
                model.load_state_dict(deepcopy(model_parameters))
                algo.set_optimizer_state(deepcopy(optimizer_state))

            log_result[name + "-proba-imp-return"] = nb_improvement_return / exp_nb_mc_estimation
            log_result[name + "-proba-imp-learning"] = nb_improvement_learning / exp_nb_mc_estimation
            log_result[name + "-avg-imp-return"] = total_improvement_return / exp_nb_mc_estimation
            log_result[name + "-avg-imp-learning"] = total_improvement_learning / exp_nb_mc_estimation

        # log the information in tensorboard
        if verbose:
            for name, value in log_result.items():
                print(name, " : ", value)

        # log the iteration number
        log_result['reinforce-iteration'] = it

        board_logger.to_log(**log_result)


if __name__ == "__main__":
    # maze version
    exp_number_ = 1

    # default learning parameter values
    nb_reinforce_interation_ = 3000
    learning_rate_ = 0.0005
    batch_size_ = 64
    learning_int_reward_weight_ = 0.
    learning_entropy_weight_ = 0.

    # default simulation experiment
    exp_nb_fit_estimation_ = 1
    exp_nb_mc_estimation_ = 100
    exp_int_reward_weight_ = 15.
    exp_entropy_weight_ = 0.

    exp_period_ = 1

    # parse the input
    parser = argparse.ArgumentParser(description="script launching experiments.")
    parser.add_argument("-e_nb", "--exp_number", type=int, default=exp_number_)

    parser.add_argument("-nb_r", "--nb_reinforce_interation", type=int, default=nb_reinforce_interation_)
    parser.add_argument("-lr", "--learning_rate", type=float, default=learning_rate_)
    parser.add_argument("-b", "--batch_size", type=int, default=batch_size_)
    parser.add_argument("-l_int", "--learning_int_reward_weight", type=float, default=learning_int_reward_weight_)
    parser.add_argument("-l_h", "--learning_entropy_weight", type=float, default=learning_entropy_weight_)

    parser.add_argument("-exp_f", "--exp_nb_fit_estimation", type=int, default=exp_nb_fit_estimation_)
    parser.add_argument("-exp_mc", "--exp_nb_mc_estimation", type=int, default=exp_nb_mc_estimation_)
    parser.add_argument("-exp_int", "--exp_int_reward_weight", type=float, default=exp_int_reward_weight_)
    parser.add_argument("-exp_h", "--exp_entropy_weight", type=float, default=exp_entropy_weight_)

    parser.add_argument("-p", "--exp_period", type=int, default=exp_period_)

    parser.add_argument("-no_stat", "--compute_no_statistics", default=False, action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=None)
    parser.add_argument("-v", "--verbose", default=False, action="store_true")

    reinforce_experiment(**vars(parser.parse_args()))
