import torch

from model.policy.determinisitic.ForwardModel import MLPModel


class KController(MLPModel):

    def __init__(self):
        super(KController, self).__init__()

        self._target = None

    def initialize(self, env, **kwargs):
        nb_inputs = env.observation_space.shape[0]
        self._target = kwargs.get("target", torch.zeros(nb_inputs)).view(1, nb_inputs)
        super(KController, self).initialize(env, layers=[], bias=False)
        return self

    def forward(self, x):
        return super(KController, self).forward(x - self._target)

    def set_k(self, value):
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.constant_(m.weight, value)

        self.net.apply(init_weights)
