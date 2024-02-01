from abc import ABC, abstractmethod

import torch


class BaseSampling(ABC):

    def __init__(self, sys, agent):
        self.sys = sys
        self.agent = agent

    @abstractmethod
    def sample(self, nb_trajectories):
        """ sample a batch of trajectories """
        raise NotImplementedError

    def cumulative_reward(self, reward_batch):
        return torch.mean(self.cumulative_reward_batch(reward_batch), dim=0).item()

    def cumulative_reward_batch(self, reward_batch):
        discount_vector = torch.empty(1, reward_batch.shape[1], 1).fill_(self.sys.gamma).cumprod(dim=1) / self.sys.gamma
        return torch.sum(reward_batch * discount_vector, dim=1)
