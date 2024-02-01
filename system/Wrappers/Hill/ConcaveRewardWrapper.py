from system.Wrappers.base import SystemWrapper


class ConcaveRewardWrapper(SystemWrapper):
    """ compute the concave envelope of the reward function """
    def __init__(self):
        super(ConcaveRewardWrapper, self).__init__()

    def initialize(self, env=None):
        self.env = env
        return self

    def concave_reward(self, state, reward):
        """ compute the new reward """
        # linear regression as a function of the position
        position, _ = state.split(1, dim=-1)

        x1, y1 = (-3, 1)
        x2, y2 = (3, 2)
        position_linear = (y2 - y1) / (x2 - x1) * (position - x1) + y1

        # indicate points on the hill
        non_concave_rewards = 1. * (position >= x1) * (position <= x2)

        # combine the linear and original rewards
        return reward * (1 - non_concave_rewards) + position_linear * non_concave_rewards

    def true_reward(self, state):
        """ compute the true reward """
        position, _ = state.split(1, dim=-1)
        return -self.env._hill(position)

    def step(self, actions):
        state, reward, terminal, info = self.env.step(actions)
        return state, self.concave_reward(state, reward), terminal, info

    def render(self, states, actions, rewards):
        return self.env.render(states, actions, self.true_reward(rewards))

    @property
    def unwrapped(self):
        return self.env.unwrapped
