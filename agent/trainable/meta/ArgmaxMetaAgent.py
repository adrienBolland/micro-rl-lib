import torch

from agent.trainable.base import TrainableAgent


class ArgmaxMetaAgent(TrainableAgent):

    def __init__(self, agent_iterable):
        super(ArgmaxMetaAgent, self).__init__()

        self._nb_actions = None

        self._agent_iterable = agent_iterable
        self._agent = None
        for a in agent_iterable:
            self._agent = a
            break

    def initialize(self, env):
        self._nb_actions = sum(env.action_space.shape)
        return self

    def get_action(self, observation):
        return self.model.get_argmax(observation)

    def get_log_likelihood(self, observation, action):
        return 1. * torch.eq(self.get_action(observation), action)

    def get_entropy(self, observation):
        return torch.zeros_like(observation.shape[:-1] + (self._nb_actions,))

    def reset_parameters(self):
        self.model.reset_parameters()

    def get_parameters(self):
        return self.model.parameters()

    def to(self, device):
        self.model.to(device)

    def _path_iterate(self, path):
        return dict()

    @property
    def model(self):
        return self._agent.model
