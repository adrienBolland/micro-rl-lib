from abc import ABC, abstractmethod


class BaseIntrinsicReward(ABC):

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

    @abstractmethod
    def initialize(self, **kwargs):
        """ Initialization fo the intrinsic reward method """
        raise NotImplementedError

    @abstractmethod
    def get_intrinsic(self, state_batch, action_batch, reward_batch):
        """ Get the intrinsic rewards """
        raise NotImplementedError
