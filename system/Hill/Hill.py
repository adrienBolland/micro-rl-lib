import torch
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from system.base import base


class Hill(base):

    def __init__(self):
        super(base, self).__init__()
        # spaces
        self._observation_space = Box(low=np.array([-4, -float("inf")], dtype=np.float32),
                                      high=np.array([5, float("inf")], dtype=np.float32),
                                      shape=(2,))
        self._action_space = Box(low=-float("inf"), high=float("inf"), shape=(1,))

        # Shape of the hill
        self.poly_coef = torch.tensor([[-0.0462963], [-0.05555556], [0.00308642], [0.00411523]])
        self.poly_power = torch.tensor([[3], [4], [5], [6]])
        self._target = 3.
        self._initial = -3.

        # mass
        self._mass = None

        # discetization
        self._discrete_time = None
        self._euler_time = None

        # tolerance
        self._tol = None

        # damping
        self._damping = 0.65

        # reset time-step
        self._time_step = 0

        # gravity
        self._g = 9.81

        # initialize the default parameters
        self.initialize()

    def initialize(self, horizon=100, device="cpu", gamma=0.99, mass=0.5, discrete_time=0.1, euler_time=0.01, tol=0.1):
        super(Hill, self).initialize(horizon, device, gamma)
        # mass
        self._mass = mass

        # discetization
        self._discrete_time = discrete_time
        self._euler_time = euler_time

        # tolerance
        self._tol = tol

        # reset time-step
        self._time_step = 0

    def reset(self, batch_size):
        """ reset the environment and return a batch of ´batch_size´ initial states of shape (´batch_size´, |S|) """
        self._time_step = 0

        initial_position = torch.empty((batch_size, 1)).fill_(self._initial)
        initial_speed = torch.zeros((batch_size, 1))

        self.state = torch.cat((initial_position, initial_speed), dim=-1)

        return self.state

    def step(self, actions):
        """ update the batch of states from a batch of actions """
        # disturb the actions
        actions = torch.normal(actions, torch.empty_like(actions).fill_(1.0))

        # new state
        self.state = self._dynamics(self.state, actions)

        # reward
        position, _ = self.state.split(1, dim=-1)
        # reward = 1. * (torch.abs(position - self._target) <= self._tol)
        # reward = -self._hill(position) - actions.abs().clamp(max=1.)
        reward = -self._hill(position)

        # update the horizon
        self._time_step += 1

        return self.state, reward, self._time_step >= self.horizon, dict()

    def close(self):
        """ close the environment """
        pass

    def render(self, states, actions, rewards):
        """ graphical view """
        plt.figure()
        plt.title("Position")
        plt.plot(states[0, :, 0].tolist())
        plt.figure()
        plt.title("Speed")
        plt.plot(states[0, :, 1].tolist())
        plt.figure()
        plt.plot(torch.clamp(actions[0, :, 0], -10, 10).tolist())
        plt.figure()
        plt.title("Reward")
        plt.plot(rewards[0, :, 0].tolist())
        plt.show()

    def to_gym(self):
        """ builds a gym environment """
        raise NotImplementedError

    def _dynamics(self, states, actions):
        position, speed = states.split(1, dim=-1)

        actions = torch.clamp(actions, -10, 10)

        for _ in range(int(self._discrete_time / self._euler_time)):
            # compute a linear approximation of the next state
            position_dot, speed_dot = self._state_derivative(position, speed, actions)

            position_new = position + position_dot * self._euler_time
            speed_new = speed + speed_dot * self._euler_time

            # clam position, if the position is clamped, the speed is set to zero
            position_new = torch.clamp(position_new, self._observation_space.low[0], self._observation_space.high[0])
            speed_new = torch.clamp(speed_new, self._observation_space.low[1], self._observation_space.high[1])

            # If an extreme position is reached, it is terminal
            terminal_position = (1. * (position <= self._observation_space.low[0])
                                 + 1. * (position >= self._observation_space.high[0]))
            position = position_new * (1 - terminal_position) + position * terminal_position
            speed = speed_new * (1 - terminal_position)

        return torch.cat((position, speed), dim=-1)

    def _state_derivative(self, position, speed, acceleration):
        p_dot = speed

        hill_dot = self._hill_dot(position)
        hill_dot2 = self._hill_dot2(position)

        s_dot = (acceleration / (self._mass * (1 + hill_dot ** 2))
                 - self._g * hill_dot / (1 + hill_dot ** 2)
                 - (speed ** 2) * hill_dot * hill_dot2 / (1 + hill_dot ** 2)
                 - self._damping * speed)

        return p_dot, s_dot

    def _hill(self, x):
        # repeat the tensor and take the power
        x_ = torch.repeat_interleave(x, repeats=4, dim=-1)
        x__ = (x ** 2) * torch.cumprod(x_, dim=-1)

        return torch.matmul(x__, self.poly_coef)

    def _hill_dot(self, x):
        # repeat the tensor and take the power
        x_ = torch.repeat_interleave(x, repeats=4, dim=-1)
        x__ = x * torch.cumprod(x_, dim=-1)

        return torch.matmul(x__, self.poly_power * self.poly_coef)

    def _hill_dot2(self, x):
        # repeat the tensor and take the power
        x_ = torch.repeat_interleave(x, repeats=4, dim=-1)
        x__ = torch.cumprod(x_, dim=-1)

        return torch.matmul(x__, self.poly_power * (self.poly_power - 1) * self.poly_coef)
