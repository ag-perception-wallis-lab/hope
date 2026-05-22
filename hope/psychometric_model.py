from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Union

import numpy as np
import parso
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen

from hope.sequential_monte_carlo import (
    IntervalTransformIndependentGaussianProposer,
)


class PsychometricModel(ABC):
    psychometric_function: Callable
    priors: Dict[str, Union[rv_frozen, multi_rv_frozen]]
    trans_prop: Optional[IntervalTransformIndependentGaussianProposer]
    bounds: Optional[Dict[str, tuple]]
    seed: Optional[int]

    @abstractmethod
    def likelihood(self, X, responses, fct_params):
        pass

    @abstractmethod
    def log_likelihood(self, X, responses, fct_params):
        pass

    @abstractmethod
    def sample_prior(self, n_samples) -> np.ndarray:
        pass

    @abstractmethod
    def log_prior(self, samples):
        pass


class BinaryPsychometricModel(PsychometricModel):
    def __init__(
        self,
        psychometric_function: Callable,
        priors: Dict[str, rv_frozen],
        bounds: Optional[
            Dict[str, tuple]
        ] = None,  # TODO if none bounds of priors will be used
        seed: Optional[int] = None,
    ):
        if bounds is not None and not set(bounds).issubset(priors):
            raise ValueError("Keys of bounds must be a subset of prior keys")
        self.priors: dict[str, rv_frozen] = priors
        self.bounds = bounds
        self._init_trans_prop()
        self.seed = seed

        self.psychometric_function = psychometric_function

    def _init_trans_prop(self):
        prior_dims = [
            np.atleast_1d(prior.rvs()).shape[0] for prior in self.priors.values()
        ]
        self.prior_dims = prior_dims
        bounded_dims = []
        dim_count = 0
        lower_bounds, upper_bounds = [], []
        for i, parameter_key in enumerate(self.priors.keys()):
            dim_count += prior_dims[i]
            if self.bounds is not None and parameter_key in self.bounds:
                lower_bound, upper_bound = self.bounds[parameter_key]
            else:
                prior = self.priors[parameter_key]
                if hasattr(prior, "support"):
                    lower_bound, upper_bound = prior.support()
                else:
                    lower_bound, upper_bound = -np.inf, np.inf
            if np.isfinite(lower_bound) and np.isfinite(upper_bound):
                if prior_dims[i] == 1:
                    bounded_dims.append(dim_count - 1)
                    lower_bounds.append(lower_bound)
                    upper_bounds.append(upper_bound)
                else:
                    raise NotImplementedError(
                        "Truncated multivariate distributions are not yet implemented"
                    )
        if len(bounded_dims) > 0:
            self.trans_prop = IntervalTransformIndependentGaussianProposer(
                bounded_dims, np.array(lower_bounds), np.array(upper_bounds)
            )

    def likelihood(self, X, responses, fct_params):
        p = self.psychometric_function(X, fct_params)
        likelihoods = np.prod(
            (np.power(p, responses) * np.power(1 - p, 1 - responses)), axis=1
        )
        likelihoods[likelihoods > 1] = 1
        likelihoods[likelihoods < 0] = 0
        return likelihoods

    def log_likelihood(self, X, responses, fct_params):
        p = self.psychometric_function(X, fct_params)
        log_likelihoods = np.sum(
            np.log(np.where(responses, p, 1) * np.where(1 - responses, 1 - p, 1)),
            axis=1,
        ).flatten()
        return log_likelihoods

    def sample_prior(self, n_samples) -> np.ndarray:
        prior_samples = np.empty((n_samples, sum(self.prior_dims)))
        if self.trans_prop:
            bounded_dims = self.trans_prop.transformed_dimensions
        current_dim = 0
        bounded_count = 0
        for i, prior in enumerate(self.priors.values()):
            new_samples = prior.rvs(size=n_samples, random_state=self.seed)
            current_dim += self.prior_dims[i]
            if current_dim in bounded_dims and self.trans_prop is not None:
                new_samples = new_samples.clip(
                    self.trans_prop.lower_bounds[bounded_count],
                    self.trans_prop.upper_bounds[bounded_count],
                )
                bounded_count += 1
            prior_samples[:, current_dim - self.prior_dims[i] : current_dim] = (
                new_samples.reshape(n_samples, -1)
                if self.prior_dims[i] > 1
                else new_samples[:, np.newaxis]
            )

        return prior_samples

    def log_prior(self, samples):
        log_prior = np.zeros(samples.shape[0])
        dims = 0
        for i, prior in enumerate(self.priors.values()):
            log_prior += prior.logpdf(
                samples[:, dims : dims + self.prior_dims[i]]
            ).flatten()
            dims += self.prior_dims[i]
        return log_prior