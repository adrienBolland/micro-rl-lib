from agent.base import BaseAgent

import torch


class ConstantActionAgent(BaseAgent):

    def __init__(self, action):
        super(ConstantActionAgent, self).__init__()
        self.nb_actions = None
        self.action = action

    def get_action(self, observation):
        return torch.empty((observation.shape[0], self.nb_actions)).fill_(self.action)

    def get_log_likelihood(self, observation, action):
        return torch.zeros((observation.shape[0], self.nb_actions))

    def initialize(self, env):
        self.nb_actions = env.action_space.shape[0]
        return self

    def save(self, path):
        """Save models"""
        pass

    def load(self, path):
        """Load models"""
        pass


class DiscreteConstantActionAgent(BaseAgent):

    def __init__(self, action):
        super(DiscreteConstantActionAgent, self).__init__()
        self.action = action

    def get_action(self, observation):
        return torch.empty((observation.shape[0], 1), dtype=torch.long).fill_(self.action)

    def get_log_likelihood(self, observation, action):
        return torch.zeros((observation.shape[0], 1))

    def initialize(self, env):
        return self

    def save(self, path):
        """Save models"""
        pass

    def load(self, path):
        """Load models"""
        pass
