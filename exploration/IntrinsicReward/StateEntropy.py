import torch
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from scipy.stats import gaussian_kde
from abc import ABC, abstractmethod

from sklearn.utils.fixes import threadpool_limits

from exploration.IntrinsicReward.base import BaseIntrinsicReward
from runner.BatchSampling import BatchSampling


class BaseStateEntropy(BaseIntrinsicReward, ABC):
    def __init__(self, env, agent):
        super(BaseStateEntropy, self).__init__(env, agent)

        # density model
        self.density_model = None

        # oracle bonus
        self.nb_samples = -1

        # chunks for memory usage
        self.max_samples_chunk = None

        # clamping
        self.min_clamp = None
        self.max_clamp = None

    def initialize(self, **kwargs):
        self.nb_samples = kwargs.get("nb_samples", -1)
        self.max_samples_chunk = kwargs.get("max_samples_chunk", float("inf"))

        self.min_clamp = kwargs.get("min_clamp", -float("inf"))
        self.max_clamp = kwargs.get("max_clamp", float("inf"))
        return self

    def get_intrinsic(self, state_batch, action_batch, reward_batch):
        # remove last state from the batch
        state_batch = state_batch[:, :-1, :]

        # build the training set in torch
        if self.nb_samples != -1 and state_batch.shape[0] < self.nb_samples:
            # resample new states
            sampler = BatchSampling(self.env, self.agent)
            oracle_state_batch, _, _ = sampler.sample(self.nb_samples - state_batch.shape[0])
            state_set = torch.cat((state_batch, oracle_state_batch), dim=0)
        elif self.nb_samples != -1 and state_batch.shape[0] > self.nb_samples:
            # subsample states
            batch_ids = np.random.choice(state_batch.shape[0], self.nb_samples, replace=False)
            state_set = state_batch[batch_ids, ...]
        else:
            state_set = state_batch

        # flatten training set to numpy
        state_set_flat = torch.flatten(state_set, 0, -2)
        discount_trajectory = torch.cumprod(torch.tensor(self.env.gamma).repeat(state_set.shape[1]), dim=0)
        discount_batch = discount_trajectory.repeat(state_set.shape[0]).unsqueeze(1)

        # build density estimation
        self._build_density_model(data=state_set_flat, weights=discount_batch)

        # flatten the evaluation set to numpy
        state_batch_flat = torch.flatten(state_batch, 0, -2)

        # compute the score
        if self.max_samples_chunk > state_batch_flat.shape[0]:
            # compute the solution as a single batch
            score = self._get_score(data=state_batch_flat)
        else:
            # divide the computation in chunks
            score = torch.cat([self._get_score(data=state_batch_flat[i:i+self.max_samples_chunk, ...])
                               for i in range(0, state_batch_flat.shape[0], self.max_samples_chunk)])

        # return density in torch
        return torch.clamp(-score.reshape(reward_batch.shape), min=self.min_clamp, max=self.max_clamp)

    @abstractmethod
    def _build_density_model(self, data, weights=None):
        raise NotImplementedError

    @abstractmethod
    def _get_score(self, data):
        raise NotImplementedError


class StateEntropyGKDE(BaseStateEntropy):
    def __init__(self, env, agent):
        super(StateEntropyGKDE, self).__init__(env, agent)

    def _build_density_model(self, data, weights=None):
        with threadpool_limits(limits=1, user_api='blas'):
            self.density_model = gaussian_kde(data.numpy().squeeze(), weights=weights.numpy().squeeze())

    def _get_score(self, data):
        with threadpool_limits(limits=1, user_api='blas'):
            score = self.density_model.logpdf(data.numpy())
        return torch.tensor(score).unsqueeze(1)


class StateEntropyGMM(BaseStateEntropy):
    def __init__(self, env, agent):
        super(StateEntropyGMM, self).__init__(env, agent)

    def _build_density_model(self, data, weights=None):
        with threadpool_limits(limits=1, user_api='blas'):
            data = data.numpy()
            weights = weights.squeeze().numpy()
            data_weights_index = np.random.choice(data.shape[0], size=data.shape[0], replace=True,
                                                  p=weights / weights.sum())
            self.density_model = GaussianMixture(n_components=10).fit(data[data_weights_index, ...])

    def _get_score(self, data):
        with threadpool_limits(limits=1, user_api='blas'):
            score = torch.tensor(self.density_model.score_samples(data.numpy()))
        return score


class StateEntropyHist1D(BaseStateEntropy):
    def __init__(self, env, agent):
        super(StateEntropyHist1D, self).__init__(env, agent)

        self.linspace_args = None

    def initialize(self, **kwargs):
        super(StateEntropyHist1D, self).initialize(**kwargs)
        self.linspace_args = kwargs.get("linspace_args", (-4, 5, 21))
        return self

    def _build_density_model(self, data, weights=None):
        min, max, nb_bins = self.linspace_args
        bins = torch.linspace(min, max, nb_bins)
        hist, _ = np.histogram(data.flatten().numpy(), bins=bins.numpy(), density=True,
                               weights=weights.flatten().numpy())
        hist = torch.tensor(hist)

        self.density_model = hist, bins

    def _get_score(self, data):
        hist, bins = self.density_model

        left_check = torch.roll(1. * (data > bins), shifts=1, dims=-1)
        right_check = 1. * (data < bins)
        bin_check = left_check[..., 1:] * right_check[..., 1:]

        prob_torch = torch.mean(hist * bin_check, dim=-1, keepdim=True)
        log_prob_torch = torch.log(1e-4 + prob_torch)

        return log_prob_torch
