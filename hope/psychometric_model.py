from abc import ABC, abstractmethod
import math
from typing import Dict

import numpy as np
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.special import expit

from .sequential_monte_carlo import WeightedParticles


class BinaryPsychometricModel(ABC):
    def __init__(self, seed=None):
        self.priors: dict[str, rv_frozen] = ...
        self.trans_prop = None # optional if transforms for distributions should be used
        self.seed = seed

    @abstractmethod
    def psychometric_fuction(self, X, fct_params):
        # Define this function in your own class!
        pass

    def likelihood(self, X, responses, fct_params):
        p = self.psychometric_function(X, fct_params)
        likelihoods = np.prod(
            (np.power(p, responses) * np.power(1 - p, 1 - responses)), axis=0
        )
        likelihoods[likelihoods > 1] = 1
        likelihoods[likelihoods < 0] = 0
        return likelihoods

    def log_likelihood(self, X, responses, fct_params):
        p = self.psychometric_function(X, fct_params)
        log_likelihoods = np.sum(
            np.log(np.where(responses, p, 1) * np.where(1 - responses, 1 - p, 1)),
            axis=1,
        )
        return log_likelihoods

    def sample_prior(self, n_samples) -> np.ndarray:
        prior_samples = []
        only_one = True
        for i, prior in enumerate(self.priors.values()):
            if i != len(self.priors) - 1:
                new_samples = prior.rvs(size=n_samples, random_state=self.seed)
                if self.trans_prop:
                    new_samples = new_samples.clip(
                        self.trans_prop.lower_bounds[i], self.trans_prop.upper_bounds[i]
                    )
                prior_samples = prior_samples + [new_samples.tolist()]
                only_one = False
            else:
                if only_one:
                    prior_samples = prior.rvs(size=n_samples, random_state=self.seed)
                    break
                prior_samples = np.array(prior_samples).T
                prior_samples = np.hstack(
                    [prior_samples, prior.rvs(size=n_samples, random_state=self.seed)]
                )
        particles = WeightedParticles(prior_samples)
        return particles


class LogisticRegressionWithLapses(BinaryPsychometricModel):
    def __init__(
        self, n_dims, seed=None
    ):  # TODO: talk to Swantje to check how to pass distributions the best
        super().__init__()
        self.n_dims = n_dims
        self.trans_prop = IntervalTransformIndependentGaussianProposer(
            [0, 1], np.array([0, 0.8]), np.array([0.2, 1])
        )
        prior = stats.multivariate_normal(
            mean=np.zeros(self.n_dims), variance=3, seed=seed
        )
        self.priors = {
            "lower_lapse": stats.beta(1, 30),
            "upper_lapse": stats.beta(1, 30),
            "weights": prior,
            "bias": stats.norm(scale=math.sqrt(3), seed=seed),
        }

    def psychometric_function(self, X, fct_params):
        if X.size == 1 and X.ndim <= 1:
            X = X.reshape((1, 1))
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        a = fct_params[:, 0]
        k = fct_params[:, 1]
        s = expit(fct_params[:, 2:] @ X.T)
        p = a + (k - a) * s
        return p

    def likelihood(self, X, responses, fct_params):
        return super().likelihood(self, X, responses, fct_params)

    def log_likelihood(self, X, responses, fct_params):
        return super().log_likelihood(self, X, responses, fct_params)

    def sample_prior(self, n_samples) -> np.ndarray:
        return super().sample_prior(self, n_samples)
