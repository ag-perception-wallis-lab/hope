from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Optional, Tuple

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from torch.distributed.flight_recorder.components.types import Op
from torch.distributions.multivariate_normal import MultivariateNormal

from .sequential_monte_carlo import n_eff
from .utils import bounded_logit, bounded_sigmoid


def adapt_proposal_width_factor(
    proposal_width_factor: float,
    acceptance_probs: ArrayLike,
    acceptance_boundaries: list[float] = [0.15, 0.5],
) -> float:
    """
    Adapts the proposal distribution's width factor if the acceptance
    probabilities are too small or too large.

    Parameters
    ----------
    proposal_width_factor: float
        The current proposal width factor.
    acceptance_probs: Iterable
        The particle MH acceptance probabilities during the last sequence of MH
        steps (for the most recent data point).
    acceptance_boundaries: list[float]
        The lower and upper boundaries for the mean acceptance probability. If the
        mean acceptance probability is below the lower boundary, the proposal width
        factor is halved. If the mean acceptance probability is above the upper
        boundary, the proposal width factor is doubled. Otherwise, it is left
        unchanged.

    Returns
    -------
    new_proposal_width_factor: float
        The adapted proposal width factor.
    """
    mean_acceptance = np.mean(acceptance_probs)
    adaptor = 1.0
    if mean_acceptance < acceptance_boundaries[0]:
        adaptor = 0.5
    if mean_acceptance > acceptance_boundaries[1]:
        adaptor = 2.0
    return adaptor * proposal_width_factor


def rule_of_thumb_bandwidths(
    locations: NDArray[np.float64],
    effective_n: Optional[int] = None,
    weights: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    """
    Assumes that the locations are i.i.d. samples or weighted. If they are
    weighted one needs to compute the actual effective number of particles. A
    problem might arise if we have weights = None or weights = 1 / N but the
    particles are acutally duplicates of each other (which usually happens after
    a importance resampling step). Then we will overestimate the effective sample
    size.

    The effective sample size can be computed as we normally do it for particle
    filters
    (https://en.wikipedia.org/wiki/Effective_sample_size#Weighted_samples).

    See
    https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
    and
    https://en.wikipedia.org/wiki/Multivariate_kernel_density_estimation#Rule_of_thumb
    for an explanation of where these formulas come frome.

    Parameters
    ----------
    locations: ndarray of shape (n_samples, n_dimensions)
        The locations of the particles.
    effective_n: int
        The effective number of particles. If None, it is computed from the weights if they are given, or set to the number of particles if weights are not given.
    weights: ndarray of shape (n_samples,)
        The weights of the particles. If None, it is assumed that all particles have equal weight.

    Returns
    -------
    bandwidths: ndarray of shape (n_dimensions,)
        The bandwidths for each dimension.
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


def propose_independent_gaussian(
    samples: NDArray[np.float64],
    proposal_vars: Optional[NDArray[np.float64]] = None,
    proposal_var_fn: Optional[Callable] = None,
    rng: Optional[np.random.Generator] = None,
) -> NDArray[np.float64]:
    """Proposes new sample locations by sampling from an independent Gaussian distribution
    centered at the current sample locations. The variances of the Gaussian can be passed directly or computed via a function (e.g. rule_of_thumb_bandwidths) that takes the current sample locations as input. If both proposal_vars and proposal_var_fn are passed, proposal_vars will be used. If neither is passed, an error is raised.

    Parameters
    ----------
    samples : ndarray of shape (n_samples, n_dimensions)
        The current locations of the particles.
    proposal_vars : Optional[NDArray[np.float64]], optional
        The variances of the Gaussian proposal distribution. If None, it will be computed using proposal_var_fn. By default None.
    proposal_var_fn : Optional[Callable], optional
        A function that computes the variances of the Gaussian proposal distribution given the current sample locations. If None, it will be ignored. By default None.
    rng : Optional[np.random.Generator], optional
        The random number generator to use. If None, a new one will be created. By default None.

    Returns
    -------
    ndarray of shape (n_samples, n_dimensions)
        The proposed new particle locations.
    """
    assert proposal_vars is not None or proposal_var_fn is not None, (
        "Either proposal_vars or proposal_var_fn must be passed."
    )
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
    samples: NDArray[np.float64],
    unnormalized_log_posterior_fn: Callable,
    proposal_fn: Callable,
    rng: Optional[np.random.Generator] = None,
    **proposal_fn_kwargs,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """
    Implements a standard Metropolis step with proposals and acceptance. For the
    Metropolis algorithm it is not possible to transform the samples into a
    different space to do proposals there since the pdf in the original space
    will not be symmetric even if it is in the transformed space (due to the
    Jacobian of the transformation). See e.g. here for an explanation:
    https://barumpark.com/blog/2019/Jacobian-Adjustments/

    Parameters
    ----------
    samples: ndarray of shape (n_samples, n_dimensions)
        The current locations of the particles.
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
    new_samples: ndarray of shape (n_accepted_samples, n_dimensions)
        The locations of the accepted proposals
    indices_of_updated_samples: ndarray of shape (n_accepted_samples,)
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
        self, samples: NDArray[np.float64], rng: Optional[np.random.Generator] = None
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
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
    samples: NDArray[np.float64],
    unnormalized_log_posterior_fn: Callable,
    trans_proposer: TransformProposer,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
    """
    Implements a standard Metropolis step with proposals and acceptance. For the
    Metropolis algorithm it is not possible to transform the samples into a
    different space to do proposals there since the pdf in the original space
    will not be symmetric even if it is in the transformed space (due to the
    Jacobian of the transformation). See e.g. here for an explanation:
    https://barumpark.com/blog/2019/Jacobian-Adjustments/

    Parameters
    ----------
    samples: ndarray of shape (n_samples, n_dimensions)
        Current locations of the particles.
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
    new_samples: ndarray of shape (n_accepted_samples, n_dimensions)
        The locations of the accepted proposals
    indices_of_updated_samples: ndarray of shape (n_accepted_samples,)
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
    """Metropolis–Hastings proposer for bounded parameters via a logit transform.

    Maps selected dimensions of the particle locations into an unbounded space
    using the logit transform, proposes new locations there
    with an independent Gaussian, maps back via the sigmoid transform, and returns the log-densities with the
    appropriate Jacobian corrections so that ``metropolis_hastings_step`` can
    compute an unbiased acceptance ratio.

    Parameters
    ----------
    transformed_dimensions : array-like of shape (n_transformed_dimensions,)
        Indices of the parameter dimensions to be transformed.
    lower_bounds : array-like of shape (n_transformed_dimensions,)
        Per-dimension lower bounds, in the same order as
        ``transformed_dimensions``.
    upper_bounds : array-like of shape (n_transformed_dimensions,)
        Per-dimension upper bounds, in the same order as
        ``transformed_dimensions``.
    proposal_vars : array-like of shape (n_transformed_dimensions,), optional
        Diagonal variances of the Gaussian proposal in the transformed space.
        Must be set before calling ``propose``; can be updated between MH
        steps (e.g. by ``adapt_proposal_width_factor``).

    Attributes
    ----------
    _transformation : Callable
        Forward transform applied to bounded dimensions before proposing
        (``bounded_sigmoid``).
    _inverse_transformation : Callable
        Inverse transform applied to proposals to recover original-space values
        (``bounded_logit``).
    """

    # transformation applied to (some dimensions of the) samples before
    # proposing
    _transformation = bounded_sigmoid
    # transformation applied to (some dimensions of the) proposals to get back
    # into original space
    _inverse_transformation = bounded_logit

    def __init__(
        self,
        transformed_dimensions: ArrayLike,
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
        proposal_vars: Optional[NDArray[np.float64]] = None,
    ):
        self.transformed_dimensions = np.asarray(transformed_dimensions, dtype=int)
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.proposal_vars = proposal_vars

    def transform(self, samples: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the logit transform to the bounded dimensions of ``samples``.

        Parameters
        ----------
        samples : ndarray of shape (..., n_dimensions)
            Particle locations in the original space.

        Returns
        -------
        ndarray of shape (..., n_dimensions)
            Copy of ``samples`` with bounded dimensions mapped to
            ``(-inf, +inf)`` via ``bounded_logit``.
        """

        _samples = np.copy(samples)
        _samples[..., self.transformed_dimensions] = bounded_logit(
            x=_samples[..., self.transformed_dimensions],
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )
        return _samples

    def backtransform(self, samples: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the sigmoid transform to map bounded dimensions back to original space.

        Parameters
        ----------
        samples : ndarray of shape (..., n_dimensions)
            Particle locations in the transformed (logit) space.

        Returns
        -------
        ndarray of shape (..., n_dimensions)
            Copy of ``samples`` with bounded dimensions mapped back to
            ``(lower_bound, upper_bound)`` via ``bounded_sigmoid``.
        """

        _samples = np.copy(samples)
        _samples[..., self.transformed_dimensions] = bounded_sigmoid(
            _samples[..., self.transformed_dimensions],
            self.lower_bounds,
            self.upper_bounds,
        )
        return _samples

    def log_jacobian_det_transform(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        a = self.lower_bounds
        b = self.upper_bounds
        return np.sum(np.log(np.abs((b - a) / ((x - a) * (b - x) + 1e-50))), axis=1)

    def propose(
        self,
        samples: NDArray[np.float64],
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Draw proposals in logit space and return Jacobian-corrected log-densities.

        Transforms ``samples`` to logit space, draws from an independent
        Gaussian centred at the transformed locations, maps proposals back to
        the original space, and computes log-densities with Jacobian corrections
        for the MH acceptance ratio.

        Parameters
        ----------
        samples : ndarray of shape (n_samples, n_dimensions)
            Current particle locations in the original space.
        rng : np.random.Generator, optional
            Random number generator. A new default generator is created if
            ``None``.

        Returns
        -------
        proposals : ndarray of shape (n_samples, n_dimensions)
            Proposed particle locations in the original space.
        proposal_logpdf : ndarray of shape (n_samples,)
            Log-density of each proposal given its sample, corrected for the
            change of variables.
        sample_logpdf : ndarray of shape (n_samples,)
            Log-density of each sample given its proposal, corrected for the
            change of variables.
        """

        if not rng:
            rng = np.random.default_rng()
        transformed_samples = self.transform(samples)
        torch.manual_seed(rng.integers(0, 2**32 - 1))
        mvn = MultivariateNormal(
            loc=torch.from_numpy(transformed_samples),
            covariance_matrix=torch.from_numpy(np.diag(self.proposal_vars)),
        )
        transformed_proposals = mvn.sample()
        proposals = self.backtransform(transformed_proposals.numpy())
        # it's the same in both directions so we can reuse it
        _logpdf = mvn.log_prob(transformed_proposals).numpy()
        proposal_logpdf = _logpdf + self.log_jacobian_det_transform(
            proposals[..., self.transformed_dimensions]
        )
        sample_logpdf = _logpdf + self.log_jacobian_det_transform(
            samples[..., self.transformed_dimensions]
        )
        return proposals, proposal_logpdf, sample_logpdf
