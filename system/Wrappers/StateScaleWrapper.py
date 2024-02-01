import torch

from system.Wrappers.base import SystemWrapper


class StateScaleWrapper(SystemWrapper):
    def __init__(self):
        super(StateScaleWrapper, self).__init__()
        self.loc = None
        self.scale = None

    def scale_state(self, unscaled_state):
        return (unscaled_state - self.loc) / self.scale

    def unscale_state(self, scaled_state):
        return (scaled_state * self.scale) + self.loc

    def initialize(self, env=None, loc=None, scale=None):
        self.env = env
        self.loc = torch.tensor(0) if loc is None else torch.tensor([loc])
        self.scale = torch.tensor(1) if scale is None else torch.tensor([scale])
        return self

    def reset(self, batch_size):
        self.env.reset(batch_size)
        return self.state

    def step(self, actions):
        _, r, done, info = self.env.step(actions)
        return self.state, r, done, info

    def render(self, states, actions, rewards):
        return self.env.render(self.unscale_state(states), actions, rewards)

    @property
    def state(self):
        return self.scale_state(self.env.state)

    @state.setter
    def state(self, s):
        self.env.state = self.unscale_state(s)

    @property
    def unwrapped(self):
        return self.env.unwrapped
