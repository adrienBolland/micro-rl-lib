from copy import deepcopy

import torch

from system.Wrappers.base import SystemWrapper


class HideStateWrapper(SystemWrapper):
    def __init__(self):
        super(HideStateWrapper, self).__init__()
        self.state_index = None

    def hide_state(self, state):
        return state[..., self.state_index]

    def extend_state(self, observation):
        # the transformation is not bijective
        state = torch.zeros(observation.shape[:-1] + (self.env.observation_space.shape[0]))
        state[..., self.state_index] = observation
        return state

    def initialize(self, env=None, state_index=()):
        self.env = env
        self._observation_space = deepcopy(self.env.observation_space)
        self._observation_space.shape = (len(state_index),)
        self.state_index = state_index
        return self

    def reset(self, batch_size):
        self.env.reset(batch_size)
        return self.state

    def step(self, actions):
        _, r, done, info = self.env.step(actions)
        return self.state, r, done, info

    @property
    def state(self):
        return self.hide_state(self.env.state)

    @property
    def observation_space(self):
        """ type of observation space """
        return self._observation_space

    @property
    def unwrapped(self):
        return self.env.unwrapped
