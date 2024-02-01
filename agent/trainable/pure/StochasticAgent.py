import os

from agent.trainable.base import TrainableAgent


class StochasticAgent(TrainableAgent):
    """Trainable agent returning deterministic actions"""

    def __init__(self, model):
        super(StochasticAgent, self).__init__()

        self._model = model

    def initialize(self, env):
        return self

    def get_action(self, observation):
        return self.model(observation).sample()

    def get_log_likelihood(self, observation, action):
        return self.model(observation).log_prob(action)

    def get_entropy(self, observation):
        return self.model(observation).entropy().unsqueeze(-1)

    def reset_parameters(self):
        self.model.reset_parameters()

    def get_parameters(self):
        return self.model.parameters()

    @property
    def model(self):
        return self._model

    def to(self, device):
        self.model.to(device)

    def _path_iterate(self, path):
        yield self.model, os.path.join(path, f'agent-save/policy')


class RecurrentStochasticAgent(StochasticAgent):
    """ Same as classical reinforce agent but observations stands for trajectories """

    def __init__(self, model):
        super(RecurrentStochasticAgent, self).__init__(model)
