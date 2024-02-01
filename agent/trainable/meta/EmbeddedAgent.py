from itertools import chain

from agent.trainable.base import TrainableAgent


class EmbeddedAgent(TrainableAgent):

    def __init__(self, agent_iterable):
        super(EmbeddedAgent, self).__init__()

        self._agent_iterable = agent_iterable
        self._agent = None
        for a in agent_iterable:
            # get last agent from the iterable
            self._agent = a

    def initialize(self, env):
        return self

    def get_action(self, observation):
        return self._agent.get_action(observation)

    def get_log_likelihood(self, observation, action):
        return self._agent.get_log_likelihood(observation, action)

    def get_entropy(self, observation):
        return self._agent.get_entropy(observation).unsqueeze(-1)

    def reset_parameters(self):
        for a in self._agent_iterable:
            a.reset_parameters()

    def get_parameters(self):
        return chain.from_iterable([a.get_parameters() for a in self._agent_iterable])

    def to(self, device):
        raise NotImplementedError()

    def _path_iterate(self, path):
        raise NotImplementedError()

    @property
    def model(self):
        raise NotImplementedError()
