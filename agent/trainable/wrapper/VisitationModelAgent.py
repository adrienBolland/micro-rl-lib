import os

from agent.trainable.base import TrainableAgent


class VisitationModelAgent(TrainableAgent):
    """Wrap a deterministic agent to make it an agent using a state visitation frequency model"""

    def __init__(self, agent, density_model):
        super(VisitationModelAgent, self).__init__()

        self._agent = agent
        self._density_model = density_model

    def initialize(self, env, **kwargs):
        return self

    def get_action(self, observation):
        return self._agent.get_action(observation)

    def get_log_likelihood(self, observation, action):
        return self._agent.get_log_likelihood(observation, action)

    def get_state_density(self, observation):
        return self._density_model(observation)

    def reset_parameters(self):
        self._density_model.reset_parameters()

    def get_parameters(self):
        return self._density_model.parameters()

    def to(self, device):
        self._agent.to(device)
        self._density_model.to(device)

    def _path_iterate(self, path):
        yield self._density_model, os.path.join(path, f'agent-save/state-density-model')

    @property
    def model(self):
        return self._density_model
