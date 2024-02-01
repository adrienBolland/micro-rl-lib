import torch

from loss.critic.base import BaseLossCritic


class CriticDPG(BaseLossCritic):
    def __init__(self, parameters):
        super(CriticDPG, self).__init__()

    def loss(self, state_batch, action_batch, reward_batch, logger=None):
        grad_action_batch = self.agent.get_action(state_batch[:, :-1, :])
        loss = -torch.mean(self.agent.get_critic(state_batch[:, :-1, :], grad_action_batch))

        return loss
