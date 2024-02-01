from model import utils
from model.policy.determinisitic.ForwardModel import MLPModel


class MLPQfunction(MLPModel):

    def __init__(self):
        super(MLPQfunction, self).__init__()
        self.nb_actions = None
        self.nb_obs = None

    def initialize(self, env, **kwargs):
        self.nb_actions = env.action_space.shape[0]
        self.nb_obs = env.observation_space.shape[0]

        self.input_size = self.nb_obs + self.nb_actions
        self.output_size = 1

        self.net, self.layers = utils.create_mlp(input_size=self.input_size, output_size=self.output_size,
                                                 layers=kwargs.get("layers", (64,)),
                                                 act_fun=kwargs.get("act_fun", "ReLU"))

        return self

    def forward(self, x):
        """ x is a state action pair """
        return self.net(x)


class MLPVfunction(MLPModel):

    def __init__(self):
        super(MLPVfunction, self).__init__()
        self.nb_actions = None
        self.nb_obs = None

    def initialize(self, env, **kwargs):
        self.nb_actions = env.action_space.shape[0]
        self.nb_obs = env.observation_space.shape[0]

        self.input_size = self.nb_obs
        self.output_size = 1

        self.net, self.layers = utils.create_mlp(input_size=self.input_size, output_size=self.output_size,
                                                 layers=kwargs.get("layers", (64,)),
                                                 act_fun=kwargs.get("act_fun", "ReLU"))

        return self

    def forward(self, x):
        """ x is a state action pair """
        return self.net(x)
