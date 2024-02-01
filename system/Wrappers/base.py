from system.base import base


class SystemWrapper(base):

    def __init__(self):
        super(SystemWrapper, self).__init__()
        self.env = None

    @classmethod
    def class_name(cls):
        return cls.__name__

    def reset(self, batch_size):
        return self.env.reset(batch_size)

    def step(self, actions):
        return self.env.step(actions)

    def close(self):
        return self.env.close()

    def render(self, states, actions, rewards):
        return self.env.render(states, actions, rewards)

    def to(self, device):
        return self.env.to(device)

    def to_gym(self):
        return self.env.to_gym()

    @property
    def state(self):
        return self.env.state

    @state.setter
    def state(self, s):
        self.env.state = s

    @property
    def horizon(self):
        """ horizon """
        return self.env.horizon

    @property
    def action_space(self):
        """ type of action space """
        return self.env.action_space

    @property
    def observation_space(self):
        """ type of observation space """
        return self.env.observation_space

    @property
    def parameter_space(self):
        """ type of parameter space """
        return self.env.parameter_space

    @property
    def gamma(self):
        """ discount factor """
        return self.env.gamma

    @property
    def unwrapped(self):
        return self.env.unwrapped
