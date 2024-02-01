import os
from abc import ABC, abstractmethod

import torch

from agent.base import BaseAgent


class TrainableAgent(BaseAgent, ABC):
    """Trainable agents are agents whose actions depends on (torch) parameters"""

    def __init__(self):
        super(TrainableAgent, self).__init__()

    @abstractmethod
    def reset_parameters(self):
        """Reset the models parameters"""
        raise NotImplementedError

    @abstractmethod
    def get_parameters(self):
        """return the models parameters"""
        raise NotImplementedError

    @abstractmethod
    def to(self, device):
        """Put the models on a device"""
        raise NotImplementedError

    def save(self, path):

        for model, path_model in self._path_iterate(path):

            dir, file = os.path.split(path_model)
            if dir != '':
                os.makedirs(dir, exist_ok=True)  # required if directory not created yet

            torch.save(model.state_dict(), path_model)

    def load(self, path):
        for model, path_model in self._path_iterate(path):
            model.load_state_dict(torch.load(path_model))

    @abstractmethod
    def _path_iterate(self, path):
        """Iterates over pairs of models and paths"""


class CriticAgent(TrainableAgent, ABC):
    def __init__(self):
        super(CriticAgent, self).__init__()

    @abstractmethod
    def get_critic(self, observation, action):
        """Returns a critic on the state-action pair"""
        raise NotImplementedError
