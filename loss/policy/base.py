from abc import ABC, abstractmethod

import torch

from loss.base import BaseLoss


class BaseLossPG(BaseLoss, ABC):
    def __init__(self):
        super(BaseLossPG, self).__init__()
        self._gamma = None

    def initialize(self, agent, **kwargs):
        self._gamma = kwargs.get("gamma", 1.)
        super(BaseLossPG, self).initialize(agent, **kwargs)
        return self

    def loss(self, state_batch, action_batch, reward_batch, logger=None):
        log_prob_a = self.agent.get_log_likelihood(state_batch[:, :-1, :], action_batch)

        with torch.no_grad():
            q = self.q_approximation(state_batch, action_batch, reward_batch)

        loss = -torch.mean(torch.sum(log_prob_a * q, dim=1))

        return loss

    @abstractmethod
    def q_approximation(self, state_batch, action_batch, reward_batch):
        raise NotImplementedError


class BaseLossRecurrentPG(BaseLoss, ABC):
    def __init__(self):
        super(BaseLossRecurrentPG, self).__init__()
        self._gamma = None

    def initialize(self, agent, **kwargs):
        self._gamma = kwargs.get("gamma", 1.)
        super(BaseLossRecurrentPG, self).initialize(agent, **kwargs)
        return self

    def loss(self, state_batch, action_batch, reward_batch, logger=None):
        log_prob_a = []

        for time in range(action_batch.shape[1]):
            trajectory = torch.cat((state_batch[:, :time+1, :].flatten(start_dim=1),
                                    action_batch[:, :time, :].flatten(start_dim=1)), dim=-1)
            log_prob_a.append(self.agent.get_log_likelihood(trajectory, action_batch[:, time]))

        log_prob_a = torch.stack(log_prob_a, dim=1)

        with torch.no_grad():
            q = self.q_approximation(state_batch, action_batch, reward_batch)

        loss = -torch.mean(torch.sum(log_prob_a * q, dim=1))

        return loss

    @abstractmethod
    def q_approximation(self, state_batch, action_batch, reward_batch):
        raise NotImplementedError
