from torch import nn
from torch.distributions import OneHotCategorical

from model import utils


class CategoricalDistModel(nn.Module):

    def __init__(self):
        super(CategoricalDistModel, self).__init__()
        self.input_size = None
        self.output_size = None
        self.layers = None
        self.net = None
        self.nb_actions = None
        self.nb_obs = None

    def initialize(self, env, **kwargs):
        self.nb_actions = env.action_space.n
        self.nb_obs = env.observation_space.shape[0]

        self.input_size = self.nb_obs
        self.output_size = self.nb_actions

        self.net, self.layers = utils.create_mlp(input_size=self.nb_obs,
                                                 output_size=self.nb_actions,
                                                 layers=kwargs.get("layers", (64,)),
                                                 act_fun=kwargs.get("act_fun", "ReLU"))

        return self

    def forward(self, x):
        return OneHotCategorical(logits=self.net(x))

    def reset_parameters(self):

        def weight_reset(m):
            if isinstance(m, nn.Linear):
                m.reset_parameters()

        self.net.apply(weight_reset)
