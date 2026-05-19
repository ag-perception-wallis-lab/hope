from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Tuple

import numpy as np
import torch
from scipy import stats
from torch.distributions.multivariate_normal import MultivariateNormal

from .sequential_monte_carlo import n_eff
from .utils import bounded_logit, bounded_sigmoid


def adapt_proposal_width_factor(
    proposal_width_factor, acceptance_probs, acceptance_boundaries=[0.15, 0.5]
):
    """
    Adapts the proposal distribution's width factor if the acceptance
    probabilities are too small or too large.

    Parameters
    ----------
    acceptance_probs: Iterable
        The particle MH acceptance probabilities during the last sequence of MH
        steps (for the most recent data point).
    """
    mean_acceptance = np.mean(acceptance_probs)
    adaptor = 1.0
    if mean_acceptance < acceptance_boundaries[0]:
        adaptor = 0.5
    if mean_acceptance > acceptance_boundaries[1]:
        adaptor = 2.0
    return adaptor * proposal_width_factor


def rule_of_thumb_bandwidths(locations, effective_n=None, weights=None):
    """
    Assumes that the locations are i.i.d. samples or weighted. If they are
    weighted one needs to compute the actual effective number of particles. A
    problem might arise if we have weights = None or weights = 1 / N but the
    particles are acutally duplicates of each other (which usually happens after
    a importance resampling step). Then we will overestimate the effctive sample
    size.

    The effective sample size can be computed as we normally do it for particle
    filters
    (https://en.wikipedia.org/wiki/Effective_sample_size#Weighted_samples).

    See
    https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
    and
    https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation#Rule_of_thumb
    for an explanation of where these formulas come frome.
    """
    if effective_n is None:
        if weights is None:
            effective_n = locations.shape[0]
        elif weights is not None:
            effective_n = n_eff(weights)
    stds = np.std(locations, axis=0)
    iqrs = stats.iqr(locations, axis=0)
    min = np.minimum(stds, iqrs / 1.34)
    n_dim = locations.shape[1]
    bandwidths = min * effective_n ** (-1 / (n_dim + 4))
    return bandwidths


def propose_isotropic_gaussian(samples, proposal_std=1.0, seed: int = None):
    # numpy can't sample from a multivariate normal in a batch fashion (although
    # in this specific case (since I'm using a diagonal covariance matrix) there
    # would have been a workaround of staying in 1D), so I turned to torch.
    if seed is not None:
        torch.set_rng_state(torch.manual_seed(seed).get_state())
    mvn = MultivariateNormal(
        loc=torch.from_numpy(samples),
        covariance_matrix=proposal_std**2 * torch.eye(samples.shape[1]),
    )
    return mvn.sample().numpy()


def propose_independent_gaussian(
    samples, proposal_vars=None, proposal_var_fn=None, rng: np.random.Generator = None
):
    assert (
        proposal_vars is not None or proposal_var_fn is not None
    ), "Either proposal_vars or proposal_var_fn must be passed."
    if not rng:
        rng = np.random.default_rng()
    if proposal_var_fn is not None:
        proposal_vars = proposal_var_fn(samples)
    torch.manual_seed(rng.integers(0, 2**32 - 1))
    mvn = MultivariateNormal(
        loc=torch.from_numpy(samples),
        covariance_matrix=torch.from_numpy(np.diag(proposal_vars)),
    )
    return mvn.sample().numpy()


def metropolis_step(
    samples: np.ndarray,
    unnormalized_log_posterior_fn: Callable,
    proposal_fn: Callable,
    rng: np.random.Generator = None,
    **proposal_fn_kwargs,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Implements a standard Metropolis step with proposals and acceptance. For the
    Metropolis algorithm it is not possible to transform the samples into a
    different space to do proposals there since the pdf in the original space
    will not be symmetric even if it is in the transformed space (due to the
    Jacobian of the transformation). See e.g. here for an explanation:
    https://barumpark.com/blog/2019/Jacobian-Adjustments/

    Parameters
    ----------
    samples: np.ndarray
        An array of samples with shape (n_samples, n_dimensions)
    unnormalized_log_posterior_fn: Callable
        Function that computes the unnormalized posterior probability density
        given a sample with dimensionality n_dimensions
    proposal_fn: Callable
        Function that proposes new sample locations given the old sample
        locations
    rng: np.random.Generator
        Random generator

    Returns
    -------
    new_samples: np.ndarray
        The locations of the accepted proposals
    indices_of_updated_samples: np.ndarray
        The indices of the samples that have an accepted proposal (i.e. that
        were updated)
    acceptance_prob: float
        The ratio of accepted proposals
    """
    proposals = proposal_fn(samples, rng=rng, **proposal_fn_kwargs)
    acceptance_probs = np.minimum(
        np.exp(
            unnormalized_log_posterior_fn(proposals)
            - unnormalized_log_posterior_fn(samples)
        ),
        np.ones(len(samples)),
    )
    accept = rng.binomial(n=1, p=acceptance_probs).reshape(-1, 1)
    indices_of_updated_samples = np.nonzero(accept.flatten())[0]
    acceptance_prob = np.sum(accept) / len(accept)
    new_samples = proposals[indices_of_updated_samples]
    return new_samples, indices_of_updated_samples, acceptance_prob


class TransformProposer(ABC):
    @abstractmethod
    def propose(
        self, samples: np.ndarray, rng: np.random.Generator = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Method to optionally transform the samples into a different space (e.g.
        log-space) to make proposals there and that returns the proposals (in
        original space) and the logpdf of the samples given the proposals and
        the proposals given the sample (while accounting for the possible
        transformation of variable and the resulting need to include the
        Jacobian of the transformation).

        Parameters
        ----------
        samples : np.ndarray
            An array of samples with shape (n_samples, n_dimensions)
        rng : np.random.Generator, optional
            Random generator, by default None

        Returns
        -------
        proposals: np.ndarray
            An array of proposals in the original space with the same shape as
            samples.
        proposal_logpdf: np.ndarray
            logpdf of the proposals given the samples (in original space; i.e.
            possibly accounting for the change of variables).
        sample_logpdf: np.ndarray
            logpdf of the samples given the proposals (in original space; i.e.
            possibly accounting for the change of variables).
        """
        pass


def metropolis_hastings_step(
    samples: np.ndarray,
    unnormalized_log_posterior_fn: Callable,
    trans_proposer: TransformProposer,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Implements a standard Metropolis step with proposals and acceptance. For the
    Metropolis algorithm it is not possible to transform the samples into a
    different space to do proposals there since the pdf in the original space
    will not be symmetric even if it is in the transformed space (due to the
    Jacobian of the transformation). See e.g. here for an explanation:
    https://barumpark.com/blog/2019/Jacobian-Adjustments/

    Parameters
    ----------
    samples: np.ndarray
        An array of samples with shape (n_samples, n_dimensions)
    unnormalized_log_posterior_fn: Callable
        Function that computes the unnormalized posterior probability density
        given a sample with dimensionality n_dimensions
    trans_proposer: TransformProposer
        Transformproposer with method propose that optionally transforms the
        samples into a different space (e.g. log-space) to make proposals there
        and that returns the proposals (in original space) and the logpdf of the
        samples given the proposals and the proposals given the sample (while
        accounting for the possible transformation of variable and the resulting
        need to include the Jacobian of the transformation).
    rng: np.random.Generator
        Random generator
    Returns
    -------
    new_samples: np.ndarray
        The locations of the accepted proposals
    indices_of_updated_samples: np.ndarray
        The indices of the samples that have an accepted proposal (i.e. that
        were updated)
    acceptance_prob: float
        The ratio of accepted proposals
    """
    if not rng:
        rng = np.random.default_rng()
    proposals, proposal_logpdf, sample_logpdf = trans_proposer.propose(samples, rng)
    acceptance_probs = np.minimum(
        np.exp(
            unnormalized_log_posterior_fn(proposals)
            - unnormalized_log_posterior_fn(samples)
            + sample_logpdf
            - proposal_logpdf
        ),
        np.ones(len(samples)),
    )
    accept = rng.binomial(n=1, p=acceptance_probs).reshape(-1, 1)
    indices_of_updated_samples = np.nonzero(accept.flatten())[0]
    acceptance_prob = np.sum(accept) / len(accept)
    new_samples = proposals[indices_of_updated_samples]
    return new_samples, indices_of_updated_samples, acceptance_prob

class IntervalTransformIndependentGaussianProposer(TransformProposer):
    """
    Transforms all the given dimenions of the samples into a given interval by
    doing a sigmoidal transformation and accounts for that in the computation of
    the logpdf of the samples and proposals. Proposals are done in an
    independent Gaussian way (in the transformed space) with variances passed at
    initialization.
    """

    # transformation applied to (some dimensions of the) samples before
    # proposing
    _transformation = bounded_sigmoid
    # transformation applied to (some dimensions of the) proposals to get back
    # into original space
    _inverse_transformation = bounded_logit

    def __init__(
        self,
        transformed_dimensions,
        lower_bounds,
        upper_bounds,
        proposal_vars=None,
    ):
        self.transformed_dimensions = transformed_dimensions
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.proposal_vars = proposal_vars

    def transform(self, samples):
        _samples = np.copy(samples)
        _samples[..., self.transformed_dimensions] = bounded_logit(
            x=_samples[..., self.transformed_dimensions],
            lower_bound=self.lower_bounds,
            upper_bound=self.upper_bounds,
        )
        return _samples

    def backtransform(self, samples):
        _samples = np.copy(samples)
        _samples[..., self.transformed_dimensions] = bounded_sigmoid(
            _samples[..., self.transformed_dimensions],
            self.lower_bounds,
            self.upper_bounds,
        )
        return _samples

    # should not be needed
    # def jacobian_det_backtransform(self, x):
    #     a = self.lower_bounds
    #     b = self.upper_bounds
    #     return np.sum(np.abs(np.exp(x) * (b - a) / (np.exp(x) + 1) ** 2), axis=1)

    def log_jacobian_det_transform(self, x):
        a = self.lower_bounds
        b = self.upper_bounds
        return np.sum(np.log(np.abs((b - a) / ((x - a) * (b - x) + 1e-50))), axis=1)

    def propose(self, samples, rng=None):
        if not rng:
            rng = np.random.default_rng()
        transformed_samples = self.transform(samples)
        torch.manual_seed(rng.integers(0, 2**32 - 1))
        mvn = MultivariateNormal(
            loc=torch.from_numpy(transformed_samples),
            covariance_matrix=torch.from_numpy(np.diag(self.proposal_vars)),
        )
        transformed_proposals = mvn.sample()
        proposals = self.backtransform(transformed_proposals)
        # it's the same in both directions so we can reuse it
        _logpdf = mvn.log_prob(transformed_proposals).numpy()
        proposal_logpdf = _logpdf + self.log_jacobian_det_transform(
            proposals[..., self.transformed_dimensions]
        )
        sample_logpdf = _logpdf + self.log_jacobian_det_transform(
            samples[..., self.transformed_dimensions]
        )
        return proposals, proposal_logpdf, sample_logpdf