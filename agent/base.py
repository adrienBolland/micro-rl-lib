from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Any agent provide actions and their loglikelihood"""

    def __init__(self):
        pass

    def __call__(self, observation):
        """Returns an action (forward)"""
        return self.get_action(observation)

    @abstractmethod
    def initialize(self, env):
        """Initializes the agent from the environment"""
        raise NotImplementedError

    @abstractmethod
    def get_action(self, observation):
        """Make an operational decision"""
        raise NotImplementedError

    @abstractmethod
    def get_log_likelihood(self, observation, action):
        """Make an operational decision"""
        raise NotImplementedError

    def get_entropy(self, observation):
        """Get entropy of the action distribution"""
        pass

    def reset_parameters(self):
        """ reset the models parameters """
        pass

    def save(self, path):
        """Save models"""
        raise NotImplementedError

    def load(self, path):
        """Load models"""
        raise NotImplementedError
