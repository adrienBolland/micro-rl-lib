import torch
import numpy as np
from gym.spaces import Box, Discrete

from system.base import base


SIMPLE_MAZE = (("W", "W", "W", "W", "W", "W", "W"),
               ("W", "I", " ", "C", " ", " ", "W"),
               ("W", " ", " ", "W", " ", " ", "W"),
               ("W", " ", "S", "W", " ", "T", "W"),
               ("W", "W", "W", "W", "W", "W", "W"))


COMPLEX_MAZE = (("W", "W", "W", "W", "W", "W", "W", "W", "W", "W", "W", "W", "W"),
                ("W", "I", " ", " ", "C", " ", " ", " ", "O", " ", " ", " ", "W"),
                ("W", " ", " ", " ", "W", " ", " ", " ", "W", " ", " ", " ", "W"),
                ("W", " ", " ", "S", "W", " ", " ", "S", "W", " ", " ", "T", "W"),
                ("W", "W", "W", "W", "W", "W", "W", "W", "W", "W", "W", "W", "W"))


def _check_position(xy_position, xy_targets):
    return torch.any(torch.all(xy_position.view(-1, 1, 2) == xy_targets.view(1, -1, 2), dim=-1), dim=-1, keepdim=True)


class MazeSwitches(base):

    def __init__(self, maze_definition=COMPLEX_MAZE, action_cost=1):
        super(MazeSwitches, self).__init__()
        # spaces
        self._observation_space = Box(low=-float("inf"), high=float("inf"), shape=(3,))
        self._action_space = Discrete(n=5)

        # time-step
        self._time_step = 0.

        # cost of moving
        self._action_cost = action_cost

        # maze definition
        self._maze_definition = np.array(maze_definition)
        self._initial_position = torch.tensor(np.argwhere(self._maze_definition == "I"))
        self._walls_position = torch.tensor(np.argwhere(self._maze_definition == "W"))
        self._switches_position = torch.tensor(np.argwhere(self._maze_definition == "S"))
        self._open_doors_position = torch.tensor(np.argwhere(self._maze_definition == "O"))
        self._closed_doors_position = torch.tensor(np.argwhere(self._maze_definition == "C"))
        self._target_position = torch.tensor(np.argwhere(self._maze_definition == "T"))

        # actions : idle = 1; down = 1; right = 2; up = 3; left = 4
        self._direction_actions = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.]])

        # initialize the default parameters
        self.initialize()

    def initialize(self, horizon=100, device="cpu", gamma=0.99):
        super(MazeSwitches, self).initialize(horizon, device, gamma)
        # reset time-step
        self._time_step = 0

        return self

    def reset(self, batch_size):
        self._time_step = 0

        self.state = torch.cat([self._initial_position.repeat(batch_size, 1), torch.zeros(batch_size, 1)], dim=-1)

        return self.state

    def step(self, actions):
        position, door_state = torch.split(self.state, [2, 1], dim=-1)
        door_state_bool = door_state.bool()

        # compute potential new position
        new_position = position + self._direction_actions[actions.squeeze().long()]

        # check if the new position is a wall
        is_wall = _check_position(new_position, self._walls_position)

        # check if the new position is a closed door
        is_closed = _check_position(new_position, self._closed_doors_position)

        # check if the new position is an open door
        is_open = _check_position(new_position, self._open_doors_position)

        # check if the new position is valid
        is_invalid_float = (is_wall + door_state_bool * is_open + (~door_state_bool) * is_closed).float()

        # compute the final new position
        new_position = is_invalid_float * position + (1 - is_invalid_float) * new_position

        # check if the new valid position is a switch
        is_switch = _check_position(new_position, self._switches_position)

        # compute the new door state
        new_door_state_bool = (~is_switch) * door_state_bool + is_switch * (~door_state_bool)
        new_door_state = new_door_state_bool.float()

        # new state
        self.state = torch.cat([new_position, new_door_state], dim=-1)

        # check if new state is target
        is_target = _check_position(new_position, self._target_position)

        # reward
        reward = -actions.clamp(min=0., max=1.) * self._action_cost + is_target.float() * 100

        # update the horizon
        self._time_step += 1

        return self.state, reward, self._time_step >= self.horizon, dict()

    def close(self):
        pass

    def render(self, states, actions, rewards):
        pass

    def to_gym(self):
        raise NotImplementedError
