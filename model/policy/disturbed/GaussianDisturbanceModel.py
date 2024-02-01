import torch
from torch import nn
from torch.distributions import MultivariateNormal


class GaussianDisturbanceModel(nn.Module):
    """ Constant Gaussian disturbance """

    def __init__(self):
        super(GaussianDisturbanceModel, self).__init__()
        self.nb_actions = None
        self.nb_obs = None

        self.std = nn.Parameter(torch.Tensor(1, 1), requires_grad=True)
        self.std_val = None

    def initialize(self, env, **kwargs):
        self.nb_actions = env.action_space.shape[0]
        self.nb_obs = env.observation_space.shape[0]

        # constant std deviation
        self.std_val = kwargs.get("std_val", 1.)
        self.reset_parameters()

        return self

    def forward(self, x):
        mean = torch.zeros(x.shape[:-1] + (self.nb_actions,))
        cov_matrix = torch.diag(torch.ones(self.nb_actions) * self.std**2).repeat(*(x.shape[:-1] + (1, 1)))
        return MultivariateNormal(mean, covariance_matrix=cov_matrix + 1e-6)

    def reset_parameters(self):
        self.set_sigma(self.std_val)

    def set_sigma(self, value):
        nn.init.constant_(self.std, value)
