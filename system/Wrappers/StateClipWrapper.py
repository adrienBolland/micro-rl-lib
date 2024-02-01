from system.Wrappers.base import SystemWrapper


class StateClipWrapper(SystemWrapper):
    """ clip the states in a given range """
    def __init__(self):
        super(StateClipWrapper, self).__init__()
        self.min = None
        self.max = None

    def scale_state(self, unscaled_state):
        return unscaled_state.clamp(min=self.min, max=self.max)

    def unscale_state(self, scaled_state):
        # the transformation is not bijective
        return scaled_state

    def initialize(self, env=None, min=-float("INF"), max=float("INF")):
        self.env = env
        self.min = min
        self.max = max
        return self

    def reset(self, batch_size):
        self.env.reset(batch_size)
        self.state = self.state  # clip
        return self.state

    def step(self, actions):
        _, r, done, info = self.env.step(actions)
        self.state = self.state  # clip
        return self.state, r, done, info

    @property
    def state(self):
        return self.unscale_state(self.env.state)

    @state.setter
    def state(self, s):
        self.env.state = self.scale_state(s)

    @property
    def unwrapped(self):
        return self.env.unwrapped
