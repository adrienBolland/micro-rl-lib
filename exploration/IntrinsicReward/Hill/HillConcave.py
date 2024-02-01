from exploration.IntrinsicReward.base import BaseIntrinsicReward


class HillConcave(BaseIntrinsicReward):

    def __init__(self, env, agent):
        super(HillConcave, self).__init__(env, agent)

    def initialize(self, **kwargs):
        return self

    def get_intrinsic(self, state_batch, action_batch, reward_batch):
        # position, _ = state_batch.split(1, dim=-1)
        position = state_batch
        return concave_reward(position[:, :-1, :], reward_batch)


def concave_reward(position, reward):
    # linear regression as a function of the position
    x1, y1 = (-3, 1)
    x2, y2 = (3, 2)
    position_linear = (y2 - y1) / (x2 - x1) * (position - x1) + y1

    # indicate points on the hill
    non_concave_rewards = 1. * (position >= x1) * (position <= x2)

    # combine the linear and original rewards
    return reward * (1 - non_concave_rewards) + position_linear * non_concave_rewards
