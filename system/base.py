from abc import ABC, abstractmethod
from collections import namedtuple
from typing import NamedTuple, Optional, Sequence, SupportsFloat

import torch

EnvSpaces = namedtuple("EnvSpaces", ["observation", "action"])


class Discrete(NamedTuple):
    n: int
    shape: Sequence[int]
    name: str = "Discrete"


class Box(NamedTuple):
    low: Optional[SupportsFloat]
    high: Optional[SupportsFloat]
    shape: Sequence[int]
    name: str = "Box"


class base(ABC):
    """ The class follows the gym prototype but allows manipulating batches of actions
    and relies on torch rather than numpy """

    def __init__(self):
        super(base, self).__init__()
        # set environment variables
        self._device = None
        self._horizon = None

        self._gamma = None

        self._action_space = None
        self._observation_space = None

        # set the internal state variable
        self._state = None

    def initialize(self, horizon=100, device="cpu", gamma=0.99):
        """ initialize the environment, all variables shall have default values """
        self._device = device
        self._horizon = horizon
        self._gamma = gamma

    @abstractmethod
    def reset(self, batch_size):
        """ reset the environment and return a batch of ´batch_size´ initial states of shape (´batch_size´, |S|) """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions):
        """ update the batch of states from a batch of actions """
        raise NotImplementedError

    @abstractmethod
    def close(self):
        """ close the environment """
        raise NotImplementedError

    def render(self, states, actions, rewards):
        """ graphical view """
        pass

    def to(self, device):
        """ put the object on a device (cpu, cuda) """
        self._device = device

        for var_name, var_ptr in vars(self).items():
            if torch.is_tensor(var_ptr):
                vars(self)[var_name] = var_ptr.to(device)

    def to_gym(self):
        """ builds a gym environment """
        raise NotImplementedError

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, s):
        self._state = s

    @property
    def horizon(self):
        """ horizon """
        return self._horizon

    @property
    def action_space(self):
        """ type of action space """
        return self._action_space

    @property
    def observation_space(self):
        """ type of observation space """
        return self._observation_space

    @property
    def spaces(self):
        return EnvSpaces(observation=self.observation_space,
                         action=self.action_space)

    @property
    def gamma(self):
        """ discount factor """
        return self._gamma

    @property
    def unwrapped(self):
        """ completely unwrap this systems """
        return self
