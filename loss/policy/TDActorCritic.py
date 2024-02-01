import torch

import loss.utils as utils
from loss.policy.base import BaseLossPG, BaseLossRecurrentPG


class TDnActorCritic(BaseLossPG):
    def __init__(self):
        super(TDnActorCritic, self).__init__()

        # number of steps in the estimate
        self.n = None

    def initialize(self, agent, **kwargs):
        self.n = kwargs.get("n", 1.)
        super(TDnActorCritic, self).initialize(agent, **kwargs)
        return self

    def q_approximation(self, state_batch, action_batch, reward_batch):
        with torch.no_grad():
            value_function_batch = self.agent.get_critic(state_batch, None)
            target, critic = utils.td_n(value_function_batch, reward_batch, self._gamma, self.n)
            td = target - critic

        return td.squeeze()


class RecurrentTDnActorCritic(BaseLossRecurrentPG):
    def __init__(self):
        super(RecurrentTDnActorCritic, self).__init__()

        # number of steps in the estimate
        self.n = None

    def initialize(self, agent, **kwargs):
        self.n = kwargs.get("n", 1.)
        super(RecurrentTDnActorCritic, self).initialize(agent, **kwargs)
        return self

    def q_approximation(self, state_batch, action_batch, reward_batch):
        with torch.no_grad():
            value_function_batch = []
            for time in range(action_batch.shape[1]):
                trajectory = torch.cat((state_batch[:, :time + 1, :].flatten(start_dim=1),
                                        action_batch[:, :time, :].flatten(start_dim=1)), dim=-1)
                value_function_batch.append(self.agent.get_critic(trajectory, None))

            value_function_batch = torch.stack(value_function_batch, dim=1)

            target, critic = utils.td_n(value_function_batch, reward_batch, self._gamma, self.n)
            td = target - critic

        return td.squeeze()


class TDActorCritic(TDnActorCritic):
    def __init__(self):
        super(TDActorCritic, self).__init__()

    def initialize(self, agent, **kwargs):
        super(TDActorCritic, self).initialize(agent, n=1, **kwargs)
        return self


class RecurrentTDActorCritic(RecurrentTDnActorCritic):
    def __init__(self):
        super(RecurrentTDActorCritic, self).__init__()

    def initialize(self, agent, **kwargs):
        super(RecurrentTDActorCritic, self).initialize(agent, n=1, **kwargs)
        return self


class A2C(TDnActorCritic):
    def __init__(self):
        super(A2C, self).__init__()

    def initialize(self, agent, **kwargs):
        super(A2C, self).initialize(agent, n=-1, **kwargs)
        return self


class RecurrentA2C(RecurrentTDnActorCritic):
    def __init__(self):
        super(RecurrentA2C, self).__init__()

    def initialize(self, agent, **kwargs):
        super(RecurrentA2C, self).initialize(agent, n=-1, **kwargs)
        return self
