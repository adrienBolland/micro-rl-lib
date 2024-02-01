import torch
from gym.spaces import Box, Discrete

from system.base import base


class Maze(base):

    def __init__(self):
        super(Maze, self).__init__()
        # spaces
        self._observation_space = Box(low=-float("inf"), high=float("inf"), shape=(1,))
        self._action_space = Discrete(n=2)

        # length of the maze
        self._length = None

        # 1 - probability to stay idle
        self._delta = None

        # reset time-step
        self._time_step = 0

        # initialize the default parameters
        self.initialize()

    def initialize(self, horizon=100, device="cpu", gamma=0.99, length=15, delta=0.3):
        super(Maze, self).initialize(horizon, device, gamma)
        # parameters
        self._length = length
        self._delta = delta

        # reset time-step
        self._time_step = 0

    def reset(self, batch_size):
        """ reset the environment and return a batch of ´batch_size´ initial states of shape (´batch_size´, |S|) """
        self._time_step = 0

        self.state = torch.empty((batch_size, 1)).fill_(1.)

        return self.state

    def step(self, actions):
        """ update the batch of states from a batch of actions """
        # disturb the actions
        actions = (2. * actions - 1.) * torch.bernoulli(torch.empty(actions.shape).fill_(self._delta))

        # check if terminal
        terminal = 1. * (self.state == self._length)

        # new state
        self.state = torch.clamp(self.state + actions, 1., self._length) * (1 - terminal) + self.state * terminal

        # reward
        reward = terminal

        # update the horizon
        self._time_step += 1

        return self.state, reward, self._time_step >= self.horizon, dict()

    def close(self):
        """ close the environment """
        pass

    def render(self, states, actions, rewards):
        """ graphical view """
        pass

    def to_gym(self):
        """ builds a gym environment """
        raise NotImplementedError
