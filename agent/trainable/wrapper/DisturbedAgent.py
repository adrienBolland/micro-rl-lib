import os

from agent.trainable.base import TrainableAgent


class DisturbedAgentWrapper(TrainableAgent):
    """Wrap a deterministic agent to make it a critic agent using a model"""

    def __init__(self, agent, dist_model):
        super(DisturbedAgentWrapper, self).__init__()

        self._agent = agent
        self._dist_model = dist_model

        self.nb_actions = None
        self.nb_obs = None

        self._recurrent_input = None
        self._recurrent_recurse = None

    def initialize(self, env, **kwargs):
        self.nb_actions = env.action_space.shape[0]
        self.nb_obs = env.observation_space.shape[0]

        self._recurrent_input = kwargs.get("recurrent_input", False)
        self._recurrent_recurse = kwargs.get("recurrent_recurse", False)
        return self

    def get_action(self, observation):
        return self._get_action_recurse(observation) + self._dist_model(observation).sample()

    def get_log_likelihood(self, observation, action):
        return self._dist_model(observation).log_prob(action - self._get_action_recurse(observation))

    def _get_action_recurse(self, observation):
        # if the recurse is not recurrent, we only keep the last state
        if self._recurrent_input and not self._recurrent_recurse:
            # observation is a trajectory (s_0, ..., s_t, a_0, ..., a_t-1)
            traj_len = (observation.shape[-1] + self.nb_actions) // (self.nb_actions + self.nb_obs)

            states, _ = observation.split((traj_len * self.nb_obs, (traj_len - 1) * self.nb_actions), dim=-1)
            _, current_state = states.split(((traj_len - 1) * self.nb_obs, self.nb_obs), dim=-1)

            observation = current_state

        return self._agent.get_action(observation)

    def get_entropy(self, observation):
        return self._dist_model(observation).entropy().unsqueeze(-1)

    def reset_parameters(self):
        self._dist_model.reset_parameters()

    def get_parameters(self):
        return self._dist_model.parameters()

    def to(self, device):
        self._agent.to(device)
        self._dist_model.to(device)

    def _path_iterate(self, path):
        yield self._dist_model, os.path.join(path, f'agent-save/disturbance')

    @property
    def model(self):
        return self._dist_model
