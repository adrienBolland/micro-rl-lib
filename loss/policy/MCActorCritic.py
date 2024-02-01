import torch

import loss.utils as utils
from loss.policy.base import BaseLossPG


class MCAdvantage(BaseLossPG):
    def __init__(self):
        super(MCAdvantage, self).__init__()

    def q_approximation(self, state_batch, action_batch, reward_batch):
        with torch.no_grad():
            value_function_batch = self.agent.get_critic(state_batch[:, :-1, :], None)
            mc = utils.mc_n(reward_batch, self._gamma, -1)
            advantage = mc - value_function_batch

        return advantage.squeeze()

    @property
    def name(self):
        return "policy"