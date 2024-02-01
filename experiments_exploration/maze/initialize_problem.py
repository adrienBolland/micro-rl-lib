import torch

import system
from agent.trainable.pure.StochasticAgent import StochasticAgent
from exploration.IntrinsicReward.Maze.MazeDense import MazeDense
from model.policy.stochastic.BinaryOpenLoopModel import BinaryOpenLoopModel
from runner.BatchSampling import BatchSampling
from exploration.IntrinsicReward.PolicyEntropy import PolicyEntropy
from exploration.IntrinsicReward.StateEntropy import StateEntropyGKDE, StateEntropyGMM, StateEntropyHist1D


OBJECTIVE_WEIGHTS = ((1., 0., 0., 0.), (1., 0.02, 0., 0.), (1., 0., 0.02, 0.), (1., 0., 0., 0.02))


def initialize_experiment():
    env = system.make('maze-v0')
    model = BinaryOpenLoopModel().initialize(env)
    agent = StochasticAgent(model).initialize(env)
    sampler = BatchSampling(env, agent)

    return env, model, agent, sampler


def compute_cumulative_reward_maze(env, stocha_agent, sampler, state_batch, action_batch, reward_batch):
    # compute intrinsic reward
    linspace_args = (0.5, env._length + 0.5, env._length + 1)
    reward_int_batch = (StateEntropyHist1D(env, stocha_agent)
                        .initialize(nb_samples=-1, max_samples_chunk=float('inf'), linspace_args=linspace_args)
                        .get_intrinsic(state_batch, action_batch, reward_batch))
    reward_dense_batch = (MazeDense(env, stocha_agent).initialize()
                          .get_intrinsic(state_batch, action_batch, reward_batch))
    reward_entropy_batch = (PolicyEntropy(env, stocha_agent).initialize()
                            .get_intrinsic(state_batch, action_batch, reward_batch))

    # compute return
    cum_reward_list = [sampler.cumulative_reward_batch(rew_batch)
                       for rew_batch in [reward_batch, reward_int_batch, reward_dense_batch, reward_entropy_batch]]

    return torch.cat(cum_reward_list, dim=-1)  # dim = (-1, 4)


def get_shaped_return(cum_reward_batch, weights=OBJECTIVE_WEIGHTS):
    tensor_shape = (1,)*(len(cum_reward_batch.shape)-1) + (-1,)
    return torch.stack([torch.sum(torch.tensor(w).view(tensor_shape) * cum_reward_batch, dim=-1)
                        for w in weights], dim=-1)
