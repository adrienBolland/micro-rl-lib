from system.Wrappers.base import SystemWrapper


class ActionClipWrapper(SystemWrapper):
    """ scale the actions played in the system """
    def __init__(self):
        super(ActionClipWrapper, self).__init__()
        self.min = None
        self.max = None

    def initialize(self, env=None, min=-float("INF"), max=float("INF")):
        self.env = env
        self.min = min
        self.max = max
        return self

    def clip_action(self, action):
        """ scale the action action """
        return action.clip(**{name: val for name, val in zip(["min", "max"], [self.min, self.max]) if val is not None})

    def step(self, actions):
        return self.env.step(self.clip_action(actions))

    def render(self, states, actions, rewards):
        return self.env.render(states, self.clip_action(actions), rewards)

    @property
    def unwrapped(self):
        return self.env.unwrapped
