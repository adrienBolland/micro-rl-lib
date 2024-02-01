import torch


def get_probability_stats(theta_tuple, direction_batch, step_size, oracle_return_list):
    # load the interpolation functions
    return_f = oracle_return_list[0]

    # evaluate pairwise the improvement
    position = torch.tensor(theta_tuple)
    average_improvement = torch.empty((2, direction_batch.shape[-1]))
    std_improvement = torch.empty((2, direction_batch.shape[-1]))
    improvement_probability = torch.empty((2, direction_batch.shape[-1]))
    for i, intrinsic_f in enumerate(oracle_return_list):
        improvement_return = return_f(position + step_size * direction_batch[:, :, i]) - return_f(position)
        improvement_intrinsic = intrinsic_f(position + step_size * direction_batch[:, :, i]) - intrinsic_f(position)

        average_improvement[0, i] = torch.mean(improvement_return)
        average_improvement[1, i] = torch.mean(improvement_intrinsic)

        std_improvement[0, i] = torch.std(improvement_return)
        std_improvement[1, i] = torch.std(improvement_intrinsic)

        improvement_probability[0, i] = torch.sum(1. * (improvement_return > 0.)) / improvement_return.shape[0]
        improvement_probability[1, i] = torch.sum(1. * (improvement_intrinsic > 0.)) / improvement_intrinsic.shape[0]

    return improvement_probability, average_improvement, std_improvement


def get_reinforce_grad(score, cum_reward, nb_reinforce_sa, max_norm, baseline=True):
    # regroup the score and trajectories by batch
    score = torch.stack(torch.split(score, nb_reinforce_sa)).unsqueeze(-1)  # shape = (-1, nb_reinforce_sa, grad_dim, 1)
    cum_reward = torch.stack(torch.split(cum_reward, nb_reinforce_sa)).unsqueeze(-2)  # shape = (-1, NB_REINFORCE_SA, 1, nb_return)

    # average reinforce estimates
    baseline = float(baseline) * torch.mean(cum_reward, dim=1, keepdim=True)
    reinforce_batch = torch.mean(score * (cum_reward - baseline), dim=1)

    # clip norm to 1
    reinforce_batch_norm = torch.norm(reinforce_batch, dim=1, keepdim=True) + 1e-18
    reinforce_batch_normalized = reinforce_batch / reinforce_batch_norm
    reinforce_batch = ((reinforce_batch_norm >= max_norm) * reinforce_batch_normalized
                       + (reinforce_batch_norm < max_norm) * reinforce_batch)

    return reinforce_batch
