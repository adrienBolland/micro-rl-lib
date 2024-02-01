import torch

from exploration.IntrinsicReward.base import BaseIntrinsicReward


class MazeDense(BaseIntrinsicReward):

    def __init__(self, env, agent):
        super(MazeDense, self).__init__(env, agent)

    def initialize(self, **kwargs):
        return self

    def get_intrinsic(self, state_batch, action_batch, reward_batch):
        return torch.sign(action_batch - 1)
