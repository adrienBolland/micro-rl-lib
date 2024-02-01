import torch
from torch import nn

from model import utils


class MLPModel(nn.Module):

    def __init__(self):
        super(MLPModel, self).__init__()

        self.input_size = None
        self.output_size = None
        self.layers = None
        self.net = None
        self.nb_actions = None
        self.nb_obs = None
        self.scale = None

    def initialize(self, env, **kwargs):
        self.output_size = self.nb_actions = env.action_space.shape[0]
        self.input_size = self.nb_obs = env.observation_space.shape[0]

        self.scale = kwargs.get("scale", None)

        self.net, self.layers = utils.create_mlp(input_size=self.input_size,
                                                 output_size=self.output_size,
                                                 layers=kwargs.get("layers", (64,)),
                                                 act_fun=kwargs.get("act_fun", "ReLU"),
                                                 bias=kwargs.get("bias", True))

        return self

    def forward(self, x):
        return self.net(x)

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
