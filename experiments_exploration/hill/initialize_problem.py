import torch

import system
from agent.trainable.pure.DeterministicAgent import DeterministicAgent
from agent.trainable.wrapper.DisturbedAgent import DisturbedAgentWrapper
from exploration.IntrinsicReward.Hill.HillConcave import HillConcave
from exploration.IntrinsicReward.PolicyEntropy import PolicyEntropy
from exploration.IntrinsicReward.StateEntropy import StateEntropyGKDE, StateEntropyGMM, StateEntropyHist1D
from model.policy.determinisitic.KController import KController
from model.policy.disturbed.GaussianDisturbanceModel import GaussianDisturbanceModel
from runner.BatchSampling import BatchSampling
from system.Wrappers.HideStateWrapper import HideStateWrapper


OBJECTIVE_WEIGHTS = ((1., 0., 0., 0.), (1., 0.05, 0., 0.), (1., 0.1, 0., 0.), (1., 1.0, 0., 0.), (1., 0., 0., 0.01),
                     (1., 0., 0., 0.1), (1., 0., 0., 0.5), (1., 0., 1., 0.))


def initialize_experiment():
    env = system.make('hill-v0')
    env = HideStateWrapper().initialize(env, [0])

    det_agent_model = KController().initialize(env, target=torch.tensor(env.unwrapped._target))
    det_agent = DeterministicAgent(det_agent_model).initialize(env)

    stocha_agent_model = GaussianDisturbanceModel().initialize(env)
    stocha_agent = DisturbedAgentWrapper(det_agent, stocha_agent_model).initialize(env,
                                                                                   recurrent_input=False,
                                                                                   recurrent_recurse=False)

    # initialize the sampling
    sampler = BatchSampling(env, stocha_agent)

    return env, det_agent_model, det_agent, stocha_agent_model, stocha_agent, sampler


def compute_cumulative_reward_hill(env, stocha_agent, sampler, state_batch, action_batch, reward_batch):
    # compute intrinsic reward
    reward_int_batch = (StateEntropyHist1D(env, stocha_agent)
                        .initialize(nb_samples=-1, max_samples_chunk=float('inf'))
                        .get_intrinsic(state_batch, action_batch, reward_batch))
    reward_concave_batch = (HillConcave(env, stocha_agent).initialize()
                            .get_intrinsic(state_batch, action_batch, reward_batch))
    reward_entropy_batch = (PolicyEntropy(env, stocha_agent).initialize()
                            .get_intrinsic(state_batch, action_batch, reward_batch))

    # compute return
    cum_reward_list = [sampler.cumulative_reward_batch(rew_batch)
                       for rew_batch in [reward_batch, reward_int_batch, reward_concave_batch, reward_entropy_batch]]

    return torch.cat(cum_reward_list, dim=-1)  # dim = (-1, 4)


def get_shaped_return(cum_reward_batch, weights=OBJECTIVE_WEIGHTS):
    tensor_shape = (1,)*(len(cum_reward_batch.shape)-1) + (-1,)
    return torch.stack([torch.sum(torch.tensor(w).view(tensor_shape) * cum_reward_batch, dim=-1)
                        for w in weights], dim=-1)
