import torch

import loss.utils as utils
from loss.critic.base import BaseLossCritic


class MCV(BaseLossCritic):
    def __init__(self):
        super(MCV, self).__init__()

        # number of steps in the estimate
        self.t_max = None

    def initialize(self, agent, **kwargs):
        self._gamma = kwargs.get("t_max", -1)
        super(MCV, self).initialize(agent, **kwargs)
        return self

    def loss(self, state_batch, action_batch, reward_batch, logger=None):
        nb_states = self.t_max
        if self.t_max < 0:
            nb_states = reward_batch.shape[1] + self.t_max + 1
        elif self.t_max > reward_batch.shape[1]:
            nb_states = reward_batch.shape[1]

        state_batch = state_batch[:, :-1, :]  # drop last state
        value_function_batch = self.agent.get_critic(state_batch[:, :nb_states, :], None)
        mc = utils.mc_n(reward_batch, self._gamma, -1)[:, :nb_states, :]

        loss = torch.mean((mc - value_function_batch)**2)

        return loss


class TDn(BaseLossCritic):
    def __init__(self):
        super(TDn, self).__init__()

        # number of steps in the estimate
        self.n = None

    def initialize(self, agent, **kwargs):
        self._gamma = kwargs.get("n", 1)
        super(TDn, self).initialize(agent, **kwargs)
        return self

    def loss(self, state_batch, action_batch, reward_batch, logger=None):
        value_function_batch = self.agent.get_critic(state_batch[:, :-1, :], action_batch)
        target, critic = utils.td_n(value_function_batch, reward_batch[:, :-1, :], self._gamma, self.n)
        td = target.detach() - critic

        loss = torch.mean(td**2)

        return loss


class RecurrentTDn(BaseLossCritic):
    def __init__(self):
        super(RecurrentTDn, self).__init__()

        # number of steps in the estimate
        self.n = None

    def initialize(self, agent, **kwargs):
        self._gamma = kwargs.get("n", 1)
        super(RecurrentTDn, self).initialize(agent, **kwargs)
        return self

    def loss(self, state_batch, action_batch, reward_batch, logger=None):
        value_function_batch = []
        for time in range(action_batch.shape[1]):
            trajectory = torch.cat((state_batch[:, :time+1, :].flatten(start_dim=1),
                                    action_batch[:, :time, :].flatten(start_dim=1)), dim=-1)
            value_function_batch.append(self.agent.get_critic(trajectory, action_batch[:, (time,), :]))

        value_function_batch = torch.stack(value_function_batch, dim=1)

        target, critic = utils.td_n(value_function_batch, reward_batch[:, :-1, :], self._gamma, self.n)
        td = target.detach() - critic

        loss = torch.mean(td**2)

        return loss


class TD(TDn):
    def __init__(self, parameters):
        super(TD, self).__init__()

    def initialize(self, agent, **kwargs):
        super(TD, self).initialize(agent, n=1, **kwargs)
        return self


class RecurrentTD(RecurrentTDn):
    def __init__(self, parameters):
        super(RecurrentTD, self).__init__()

    def initialize(self, agent, **kwargs):
        super(RecurrentTD, self).initialize(agent, n=1, **kwargs)
        return self
