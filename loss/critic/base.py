from abc import ABC

from loss.base import BaseLoss


class BaseLossCritic(BaseLoss, ABC):
    def __init__(self):
        super(BaseLossCritic, self).__init__()
        self._gamma = None

    def initialize(self, agent, **kwargs):
        self._gamma = kwargs.get("gamma", 1.)
        super(BaseLossCritic, self).initialize(agent, **kwargs)
        return self
