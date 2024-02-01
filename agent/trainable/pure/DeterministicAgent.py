import os

import torch

from agent.trainable.base import TrainableAgent


class DeterministicAgent(TrainableAgent):
    """Trainable agent returning deterministic actions"""

    def __init__(self, model):
        super(DeterministicAgent, self).__init__()

        self._model = model
        self._nb_actions = None

    def initialize(self, env):
        self._nb_actions = sum(env.action_space.shape)
        return self

    def get_action(self, observation):
        return self.model(observation)

    def get_log_likelihood(self, observation, action):
        return 1. * torch.eq(self.get_action(observation), action)

    def get_entropy(self, observation):
        return torch.zeros_like(observation.shape[:-1] + (self._nb_actions, 1))

    def reset_parameters(self):
        self.model.reset_parameters()

    def get_parameters(self):
        return self.model.parameters()

    @property
    def model(self):
        return self._model

    def to(self, device):
        self.model.to(device)

    def _path_iterate(self, path):
        yield self.model, os.path.join(path, f'agent-save/policy')


class RecurrentDeterministicAgent(DeterministicAgent):
    """ Same as classical reinforce agent but observations stands for trajectories """

    def __init__(self, model):
        super(RecurrentDeterministicAgent, self).__init__(model)
