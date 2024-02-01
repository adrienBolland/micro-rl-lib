import torch
from torch import nn
from torch.distributions import MultivariateNormal

from model import utils


class GaussianDistModel(nn.Module):

    def __init__(self):
        super(GaussianDistModel, self).__init__()
        self.input_size = None
        self.output_size = None
        self.layers = None
        self.net = None
        self.nb_actions = None
        self.nb_obs = None
        self.correlated = None
        self.id_lower = None
        self.scale = None

    def initialize(self, env, **kwargs):
        self.nb_actions = env.action_space.shape[0]
        self.nb_obs = env.observation_space.shape[0]

        self.correlated = kwargs.get("correlated", False)
        self.id_lower = torch.tril_indices(self.nb_actions, self.nb_actions, -1)

        self.scale = kwargs.get("scale", None)

        self.input_size = self.nb_obs
        self.output_size = self.nb_actions * (self.nb_actions + 3) // 2 if self.correlated else 2 * self.nb_actions

        self.net, self.layers = utils.create_mlp(input_size=self.input_size,
                                                 output_size=self.output_size,
                                                 layers=kwargs.get("layers", (64,)),
                                                 act_fun=kwargs.get("act_fun", "ReLU"))

        return self

    def forward(self, x):
        mean, covariance_matrix = self.get_paramters(x)
        constant = 10e-6 * torch.diag_embed(torch.ones_like(mean))
        return MultivariateNormal(mean, covariance_matrix=covariance_matrix + constant)

    def get_paramters(self, x):
        features = self.net(x)
        mean, std, cov = features.split((self.nb_actions, self.nb_actions, self.output_size - 2 * self.nb_actions),
                                        dim=-1)

        scale_tril = torch.diag_embed(std.abs())

        if self.correlated:
            scale_tril[..., self.id_lower[0, :], self.id_lower[1, :]] = cov

        covariance_matrix = torch.matmul(scale_tril, scale_tril.transpose(-2, -1))

        return mean, covariance_matrix

    def get_argmax(self, x):
        mean, _ = self.get_paramters(x)
        return mean

    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        self.net.apply(weight_reset)

        # scale the last layer by a factor
        if self.scale is not None:
            with torch.no_grad():
                last_layer_w = self.layers[-1].weight
                last_layer_w.copy_(last_layer_w / self.scale)
