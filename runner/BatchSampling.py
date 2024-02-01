import torch

from runner.base import BaseSampling


class BatchSampling(BaseSampling):

    def __init__(self, sys, agent):
        super(BatchSampling, self).__init__(sys, agent)

    def sample(self, nb_trajectories):
        """ sample a batch of trajectories (the system implements torch-batch step) """
        done = False

        state_batch = []
        reward_batch = []
        action_batch = []

        state = self.sys.reset(batch_size=nb_trajectories)
        state_batch.append(state)

        while not done:
            action = self.agent(state)
            state, reward, done, _ = self.sys.step(action)

            state_batch.append(state)
            reward_batch.append(reward)
            action_batch.append(action)

        return [torch.stack(batch, dim=1) for batch in [state_batch, action_batch, reward_batch]]


class RecurrentBatchSampling(BaseSampling):
    """The agent is now trajectory dependent"""

    def __init__(self, sys, agent):
        super(RecurrentBatchSampling, self).__init__(sys, agent)

    def sample(self, nb_trajectories):
        done = False

        state_batch = torch.empty((nb_trajectories, self.sys.horizon+1, sum(self.sys.observation_space.shape)))
        reward_batch = torch.empty((nb_trajectories, self.sys.horizon, 1))
        action_batch = torch.empty((nb_trajectories, self.sys.horizon, sum(self.sys.action_space.shape)))

        time = 0
        state_batch[:, time, :] = self.sys.reset(batch_size=nb_trajectories)

        while not done:
            trajectory = torch.cat((state_batch[:, :time+1, :].flatten(start_dim=1),
                                    action_batch[:, :time, :].flatten(start_dim=1)), dim=-1)

            action = self.agent(trajectory)
            state, reward, done, _ = self.sys.step(action)

            state_batch[:, time+1, :] = state
            reward_batch[:, time, :] = reward
            action_batch[:, time, :] = action

            time += 1

        return state_batch, action_batch, reward_batch
