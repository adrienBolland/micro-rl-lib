import torch


def td_n(value_function_batch, reward_batch, gamma, n):
    # faster computation if n is one
    if n == 1:
        return _td_1(value_function_batch, reward_batch, gamma)

    # clip n
    if n < 0:
        n = reward_batch.shape[1] + n + 1
    elif n > reward_batch.shape[1]:
        n = reward_batch.shape[1]

    # faster computation if n is T
    if n == reward_batch.shape[1]:
        return _td_max(value_function_batch, reward_batch, gamma)

    # compute the monte carlo rollout
    sum_r_n = mc_n(reward_batch, gamma, n)

    # bootstrap
    nb_elements = sum_r_n.shape[1] - (n-1)
    value_next = torch.empty(reward_batch.shape)

    value_next[:, :nb_elements, :] = value_function_batch[:, n:, :]

    if n > 1:
        value_next[:, nb_elements:, :] = value_next[:, (nb_elements-1,), :]
        discount_vector = torch.vander(torch.tensor([gamma]), N=n).reshape(1, n, 1)
        value_next[:, nb_elements:, :] = discount_vector[:, :-1, :] * value_next[:, (nb_elements-1,), :]

    value_next[:, :nb_elements, :] = gamma ** n * value_next[:, :nb_elements, :]

    return sum_r_n + value_next, value_function_batch[:, :-1]


def _td_1(value_function_batch, reward_batch, gamma):
    return reward_batch + gamma * value_function_batch[:, 1:], value_function_batch[:, :-1]


def _td_max(value_function_batch, reward_batch, gamma):
    # add a last reward equal to the value function of the last state
    extended_reward = torch.cat((reward_batch, value_function_batch[:, (-1,), :]), dim=1)

    # sum all rewards to go
    discount_vector = torch.empty(1, extended_reward.shape[1], 1).fill_(gamma).cumprod(dim=1) / gamma
    partial_cum_r = flip_cumsum(extended_reward * discount_vector) / discount_vector

    return partial_cum_r[:, :-1], value_function_batch[:, :-1]


def mc_n(reward_batch, gamma, n):
    # clip n
    if n < 0:
        n = reward_batch.shape[1] + n + 1
    elif n > reward_batch.shape[1]:
        n = reward_batch.shape[1]

    discount_vector = torch.vander(torch.tensor([gamma]), N=n, increasing=True).reshape(1, 1, n)
    sum_r_n = torch.nn.functional.conv1d(reward_batch.swapaxes(1, 2), discount_vector, padding=n-1).swapaxes(1, 2)
    sum_r_n = sum_r_n[:, n-1:, :]

    return sum_r_n


def flip_cumsum(tensor, dim=1):
    return torch.cumsum(tensor.flip(dim), dim=dim).flip(dim)
