import os
from abc import ABC, abstractmethod

import torch

from agent.trainable.base import CriticAgent


class CriticAgentWrapperBase(CriticAgent, ABC):

    def __init__(self, agent, critic_model):
        super(CriticAgentWrapperBase, self).__init__()

        self._agent = agent
        self._critic_model = critic_model

    def initialize(self, env):
        return self

    def get_action(self, observation):
        return self._agent.get_action(observation)

    def get_log_likelihood(self, observation, action):
        return self._agent.get_log_likelihood(observation, action)

    def get_entropy(self, observation):
        return self._agent.get_entropy(observation).unsqueeze(-1)

    @abstractmethod
    def get_critic(self, observation, action):
        raise NotImplementedError

    def reset_parameters(self):
        self._critic_model.reset_parameters()

    def get_parameters(self):
        return self._critic_model.parameters()

    def to(self, device):
        self._agent.to(device)
        self._critic_model.to(device)

    def _path_iterate(self, path):
        yield self._critic_model, os.path.join(path, f'agent-save/critic')

    @property
    def model(self):
        return self._critic_model


class QCriticAgentWrapper(CriticAgentWrapperBase):

    def __init__(self, agent, critic_model):
        super(QCriticAgentWrapper, self).__init__(agent, critic_model)

    def get_critic(self, observation, action):
        return self._critic_model(torch.cat((observation, action), dim=-1))


class VCriticAgentWrapper(CriticAgentWrapperBase):

    def __init__(self, agent, critic_model):
        super(VCriticAgentWrapper, self).__init__(agent, critic_model)

    def get_critic(self, observation, action):
        return self._critic_model(observation)
