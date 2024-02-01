import torch

import loss.utils as utils
from loss.policy.base import BaseLossPG


class GAEActorCritic(BaseLossPG):
    def __init__(self):
        super(GAEActorCritic, self).__init__()

        # number of steps in the estimate
        self.n = None

        # self discount value
        self.gae_lambda = None

    def initialize(self, agent, **kwargs):
        self.n = kwargs.get("n", 1.)
        self.gae_lambda = kwargs.get("gae_lambda", 0.9)
        super(GAEActorCritic, self).initialize(agent, **kwargs)
        return self

    def q_approximation(self, state_batch, action_batch, reward_batch):
        with torch.no_grad():
            # compute the temporal differences TD(1)
            value_function_batch = self.agent.get_critic(state_batch, None)
            target_td1, critic_td1 = utils.td_n(value_function_batch, reward_batch, self._gamma, 1)
            td1 = target_td1 - critic_td1

            # compute the discounted sum of TD(1)
            gae = utils.mc_n(td1, self._gamma * self.gae_lambda, self.n)

        return gae.squeeze()
