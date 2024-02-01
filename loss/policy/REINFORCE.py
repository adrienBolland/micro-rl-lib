import torch

from loss.policy.base import BaseLossPG, BaseLossRecurrentPG


class Reinforce(BaseLossPG):
    def __init__(self):
        super(Reinforce, self).__init__()

    def q_approximation(self, state_batch, action_batch, reward_batch):
        return _q_approximation(reward_batch, self._gamma)

    @property
    def name(self):
        return "policy"


class RecurrentReinforce(BaseLossRecurrentPG):
    def __init__(self):
        super(RecurrentReinforce, self).__init__()

    def q_approximation(self, state_batch, action_batch, reward_batch):
        return _q_approximation(reward_batch, self._gamma)

    @property
    def name(self):
        return "policy"


def _q_approximation(reward_batch, gamma):
    discount_vector = torch.empty(1, reward_batch.shape[1], 1).fill_(gamma).cumprod(dim=1) / gamma
    cum_r = torch.sum(reward_batch * discount_vector, dim=1)
    mean_cum_r = torch.mean(cum_r).item()
    return cum_r - mean_cum_r
