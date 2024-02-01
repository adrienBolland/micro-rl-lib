from abc import abstractmethod, ABC


class BaseLoss(ABC):

    def __init__(self):
        super(BaseLoss, self).__init__()
        self._agent = None

    def __call__(self, state_batch, action_batch, reward_batch, logger=None):
        return self.loss(state_batch, action_batch, reward_batch, logger)

    def initialize(self, agent, **kwargs):
        # adam optimizer parameters
        self._agent = agent

        return self

    @abstractmethod
    def loss(self, state_batch, action_batch, reward_batch, logger=None):
        """Compute the loss"""
        raise NotImplementedError

    @property
    def agent(self):
        return self._agent
