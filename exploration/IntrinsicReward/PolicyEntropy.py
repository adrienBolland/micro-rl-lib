from exploration.IntrinsicReward.base import BaseIntrinsicReward


class PolicyEntropy(BaseIntrinsicReward):

    def __init__(self, env, agent):
        super(PolicyEntropy, self).__init__(env, agent)

        # oracle bonus
        self.reward_based = None

    def initialize(self, **kwargs):
        self.reward_based = kwargs.get("reward_based", True)
        return self

    def get_intrinsic(self, state_batch, action_batch, reward_batch):
        """ Get the intrinsic rewards and weights """
        if self.reward_based:
            return -self.agent.get_log_likelihood(state_batch[:, :-1, :], action_batch).reshape(reward_batch.shape)
        else:
            return -self.agent.get_entropy(state_batch[:, :-1, :]).unsqueeze(-1)
