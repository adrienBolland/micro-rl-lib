import torch

from system.Wrappers.base import SystemWrapper


class OHEWrapper(SystemWrapper):
    def __init__(self):
        super(OHEWrapper, self).__init__()
        self.num_to_ohe_tensor = None
        self.ohe_to_num_tensor = None

    def initialize(self, env=None, *args, **kwargs):
        self.env = env
        self.num_to_ohe_tensor = torch.FloatTensor(1, self.env.action_space.n).zero_()
        self.ohe_to_num_tensor = torch.arange(start=0, end=self.env.action_space.n).reshape(-1, 1).float()
        return self

    def num_to_ohe(self, num_action):
        return self.num_to_ohe_tensor.repeat(num_action.shape[0], 1).scatter(1, num_action, 1)

    def ohe_to_num(self, ohe_action):
        return torch.matmul(ohe_action, self.ohe_to_num_tensor).long()

    def step(self, actions):
        return self.env.step(self.ohe_to_num(actions))

    def render(self, states, actions, rewards):
        return self.env.render(states, self.ohe_to_num(actions), rewards)

    def to(self, device):
        """ put the object on a device (cpu, cuda) """
        self.num_to_ohe_tensor = self.num_to_ohe_tensor.to(device)
        self.ohe_to_num_tensor = self.ohe_to_num_tensor.to(device)
        self.env.to(device)

    @property
    def unwrapped(self):
        return self.env.unwrapped
