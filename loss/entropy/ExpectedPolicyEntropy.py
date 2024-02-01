import torch

from loss.base import BaseLoss


class ExpectedPolicyEntropy(BaseLoss):

    def loss(self, state_batch, action_batch, reward_batch, logger=None):
        return torch.mean(self.agent.get_entropy(state_batch))
