from agent.base import BaseAgent

import math
import torch


class DiscreteRandomAgent(BaseAgent):

    def __init__(self):
        super(DiscreteRandomAgent, self).__init__()
        self.nb_actions = None

    def get_action(self, observation):
        return torch.randint(0, self.nb_actions, (observation.shape[0], 1))

    def get_log_likelihood(self, observation, action):
        return torch.log(torch.empty_like(action).fill_(1. / self.nb_actions))

    def initialize(self, env, **kwargs):
        self.nb_actions = env.action_space.n
        return self

    def save(self, path):
        """Save models"""
        pass

    def load(self, path):
        """Load models"""
        pass


class NormalRandomAgent(BaseAgent):

    def __init__(self):
        super(NormalRandomAgent, self).__init__()
        self.nb_actions = None

    def get_action(self, observation):
        return torch.randn((observation.shape[0], self.nb_actions))

    def get_log_likelihood(self, observation, action):
        return 0.25 * action * self.nb_actions**2 * math.log(2*math.pi)

    def initialize(self, env):
        self.nb_actions = env.action_space.shape[0]
        return self

    def save(self, path):
        """Save models"""
        pass

    def load(self, path):
        """Load models"""
        pass
