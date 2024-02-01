import torch
from torch import nn


class BinaryOpenLoopModel(nn.Module):

    def __init__(self):
        super(BinaryOpenLoopModel, self).__init__()
        self._theta = nn.Parameter(torch.tensor([0.]))

    def initialize(self, env, **kwargs):
        return self

    def forward(self, x):
        with torch.no_grad():
            proba = torch.ones_like(x)

        return torch.distributions.Bernoulli(probs=proba * self._theta)

    def reset_parameters(self):
        self.set_theta(0.)

    def set_theta(self, theta):
        nn.init.constant_(self._theta, theta)
