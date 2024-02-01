from itertools import chain

import torch
from torch.optim import Adam

from algo.base import BaseAlgo
from algo import utils
from loss.policy.REINFORCE import Reinforce as ReinforceLoss
from loss.entropy.ExpectedPolicyEntropy import ExpectedPolicyEntropy


class Reinforce(BaseAlgo):

    def __init__(self, env, sampler, agent_dict, logger=None):
        super(Reinforce, self).__init__(env, sampler, agent_dict, logger)

        # losses used by the algo
        self.reinforce_loss = ReinforceLoss().initialize(self.agent_dict["policy"])
        self.entropy_loss = ExpectedPolicyEntropy().initialize(self.agent_dict["policy"])

        # optimized parameters
        self.parameters = list(chain.from_iterable([self.agent_dict[n].get_parameters()
                                                    for n in self.agent_optim_list]))

        # parameters to be initialized
        self.optimizer_parameters = None
        self.batch_size = None
        self.clip_norm = None
        self.clip_max = None
        self.entropy_weight = None
        self.optimizer = None
        self.nb_it = 0.
        self.int_reward_function_list = None
        self.int_reward_weights = None

    def initialize(self, **kwargs):
        # adam optimizer parameters
        self.optimizer_parameters = kwargs.get("optimizer_parameters", {"lr": 0.001})

        # batch size
        self.batch_size = kwargs.get("batch_size", 64)

        # clip norm or value of the gradient
        self.clip_norm = kwargs.get("clip_norm", None)
        self.clip_max = kwargs.get("clip_max", None)

        # entropy weight
        self.entropy_weight = kwargs.get("entropy_weight", 0.)

        # intrinsic reward functions
        self.int_reward_function_list = kwargs.get("shaper_list", [])
        self.int_reward_weights = kwargs.get("weight_list", [])

        # initialize the optimizer
        self.optimizer = Adam(self.parameters, **self.optimizer_parameters)

        # iteration counter
        self.nb_it = 0

        return self

    def fit(self, nb_iterations):

        for it in range(nb_iterations):
            # reset gradients
            self.optimizer.zero_grad()

            # sample trajectories
            with torch.no_grad():
                state_batch, action_batch, reward_batch = self.sampler.sample(self.batch_size)

                if len(self.int_reward_function_list):
                    list_int_reward = [shaper.get_intrinsic(state_batch, action_batch, reward_batch)
                                       for shaper in self.int_reward_function_list]
                    weighted_rewards_batch = torch.cat([reward * weight
                                                        for reward, weight in zip(list_int_reward,
                                                                                  self.int_reward_weights)], dim=-1)
                    int_reward = torch.sum(weighted_rewards_batch, dim=-1, keepdim=True)

                else:
                    list_int_reward = []
                    int_reward = torch.zeros_like(reward_batch)

            # compute the losses
            loss_policy = self.reinforce_loss(state_batch, action_batch, reward_batch + int_reward, self.logger)
            loss_entropy = self.entropy_loss(state_batch, action_batch, reward_batch + int_reward, self.logger)

            # perform the optimization
            loss = loss_policy - self.entropy_weight * loss_entropy
            loss.backward()

            if self.clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters, max_norm=self.clip_norm)

            if self.clip_max is not None:
                torch.nn.utils.clip_grad_value_(self.parameters, clip_value=self.clip_max)

            self.optimizer.step()

            # manage the logs
            if self.logger is not None:
                with torch.no_grad():
                    self.logger.to_log(**{"iteration-number": it,
                                          "total-iteration-number": self.nb_it,
                                          "return": self.sampler.cumulative_reward(reward_batch),
                                          "learning-obj": self.sampler.cumulative_reward(reward_batch + int_reward),
                                          "loss-reinforce": loss_policy.item(),
                                          "loss-entropy": loss_entropy.item(),
                                          "loss-total": loss.item(),
                                          "grad-norm": utils.gradient_norm(self.parameters),
                                          "parameters": self.parameters,
                                          "optim-state-dict": self.optimizer.state_dict(),
                                          "trajectory": [state_batch, action_batch, reward_batch],
                                          "intrinsic-rewards": list_int_reward})

            self.nb_it += 1

    def set_entropy_weight(self, weight):
        self.entropy_weight = weight

    def set_reward_shaping(self, shaper_list, weight_list):
        self.int_reward_function_list = shaper_list
        self.int_reward_weights = weight_list

    def set_optimizer_state(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    @property
    def agent_optim_list(self):
        yield "policy"
