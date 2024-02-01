from abc import ABC, abstractmethod


class BaseAlgo(ABC):

    def __init__(self, env, sampler, agent_dict, logger):
        self.env = env
        self.sampler = sampler
        self.agent_dict = agent_dict
        self.logger = logger

    @abstractmethod
    def initialize(self, **kwargs):
        """ Parameters for performing the optimization """
        raise NotImplementedError

    @abstractmethod
    def fit(self, **kwargs):
        """ Execute the optimization """
        raise NotImplementedError
