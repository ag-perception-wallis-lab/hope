from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.stats._distn_infrastructure import rv_frozen
from scipy.stats._multivariate import multi_rv_frozen

from hope.sequential_monte_carlo import (
    IntervalTransformIndependentGaussianProposer,
)


class PsychometricModel(ABC):
    """Abstract base class for psychometric models used in sequential Monte Carlo inference.

    Subclasses must implement ``likelihood`` and ``log_likelihood`` for a specific
    response type (e.g. binary, ordinal). They should also set ``psychometric_function``
    and ``priors`` as instance attributes, typically in ``__init__``.

    Attributes
    ----------
    psychometric_function : Callable
        Maps stimuli and parameter arrays to predicted response probabilities.
    priors : Dict[str, rv_frozen | multi_rv_frozen]
        Named priors for each psychometric function parameter. Keys must match
        the parameter names used elsewhere (e.g. in ``bounds``).
    prior_dims : list[int]
        Dimensionality of each prior, in the same order as ``priors``.
    trans_prop : IntervalTransformIndependentGaussianProposer, optional
        Interval-transform proposer for bounded parameters. Set automatically
        by subclasses based on ``bounds`` and prior supports.
    bounds : Dict[str, tuple], optional
        Hard lower and upper bounds for individual parameters, keyed by the
        same names as ``priors``. Parameters without explicit bounds fall back
        to the support of their prior.
    seed : int, optional
        Random seed forwarded to ``prior.rvs`` calls.
    """

    psychometric_function: Callable
    priors: Dict[str, Union[rv_frozen, multi_rv_frozen]]
    prior_dims: list[int]
    trans_prop: Optional[IntervalTransformIndependentGaussianProposer] = None
    bounds: Optional[Dict[str, tuple]]
    seed: Optional[int]

    @abstractmethod
    def likelihood(self, x, response, fct_params):
        pass

    @abstractmethod
    def log_likelihood(self, X, responses, fct_params):
        pass

    def sample_prior(self, n_samples) -> np.ndarray:
        """Draw samples from the joint prior over all parameters.

        Samples from each prior are concatenated along the last axis to form
        a flat parameter vector. For bounded parameters samples are clipped to ``[lower_bound, upper_bound]``
        before being stored.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        ndarray of shape (n_samples, n_parameters)
            Each row is a joint prior sample, with parameters ordered as in
            ``priors``.
        """

        prior_samples = np.empty((n_samples, sum(self.prior_dims)))
        if self.trans_prop:
            bounded_dims = self.trans_prop.transformed_dimensions
        current_dim = 0
        bounded_count = 0
        for i, prior in enumerate(self.priors.values()):
            new_samples = prior.rvs(size=n_samples, random_state=self.seed)
            current_dim += self.prior_dims[i]
            if self.trans_prop is not None and current_dim in bounded_dims:
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

    def log_prior(self, fct_params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Evaluates the log-prior probability of the given parameters by summing the log-probabilities of the individual priors for each parameter.

        Parameters
        ----------
        fct_params : ndarray of shape (n_particles, n_parameters)
             A 2D array where each row corresponds to a set of parameters (e.g. location of one particle) for the psychometric function.

        Returns
        -------
        ndarray of shape (n_particles,)
            A 1D array where each element corresponds to the log-prior probability of the respective set of parameters in fct_params.
        """
        log_prior = np.zeros(fct_params.shape[0])
        dims = 0
        for i, prior in enumerate(self.priors.values()):
            log_prior += prior.logpdf(
                fct_params[:, dims : dims + self.prior_dims[i]]
            ).flatten()
            dims += self.prior_dims[i]
        return log_prior


class BinaryPsychometricModel(PsychometricModel):
    """Psychometric model for binary (0/1) responses.

    Assumes a Bernoulli observation model: the probability of a '1' response is
    given by ``psychometric_function(x, fct_params)``. Bounded parameters are
    handled via an transformed proposal distribution, which is automatically initialized based on the provided bounds and prior supports.

    Parameters
    ----------
    psychometric_function : Callable
        A function ``f(X, fct_params) -> ndarray of shape (n_trials, n_particles)``
        that returns the probability of a '1' response for each (trial, particle)
        combination.
    priors : Dict[str, rv_frozen]
        Named priors for each psychometric function parameter. Keys are
        used to match entries in ``bounds``.
    bounds : Dict[str, tuple], optional
        Hard ``(lower, upper)`` bounds for individual parameters. Keys must be a
        subset of ``priors`` keys. Parameters not listed here fall back to the
        support of their prior. If neither ``bounds`` nor the prior support is
        finite, no interval transform is applied to that parameter.
    seed : int, optional
        Random seed.

    Raises
    ------
    ValueError
        If any key in ``bounds`` is not present in ``priors``.
    NotImplementedError
        If a bounded parameter has ``prior_dims > 1`` (truncated multivariate
        distributions are not yet supported).
    """

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
        """Initialise ``trans_prop`` from ``bounds`` and prior supports.

        Iterates over all parameters in ``priors`` and determines whether each
        one is bounded (either via ``self.bounds`` or the prior's own
        support). Bounded scalar parameters are collected and used to
        construct an ``IntervalTransformIndependentGaussianProposer``, which is
        stored as ``self.trans_prop``. If no parameters are bounded,
        ``self.trans_prop`` is left as ``None``.
        """

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

    def likelihood(
        self,
        x: NDArray[np.float64],
        response: np.float64,
        fct_params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute the Bernoulli likelihood of a single trial.

        Evaluates ``p^response * (1 - p)^(1 - response)`` for each particle,
        where ``p = psychometric_function(x, fct_params)``. For multiple trials,
        use ``log_likelihood`` instead to avoid numerical underflow.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Stimulus presented in the trial.
        response : float
            Observed binary response (0 or 1).
        fct_params : ndarray of shape (n_particles, n_parameters)
            Each row is a parameter vector for one particle.

        Returns
        -------
        ndarray of shape (n_particles,)
            Bernoulli likelihood of the observed response for each particle.
        """

        p = self.psychometric_function(x, fct_params)
        likelihoods = np.prod(
            (np.power(p, response) * np.power(1 - p, 1 - response)), axis=1
        )
        likelihoods[likelihoods > 1] = 1
        likelihoods[likelihoods < 0] = 0
        return likelihoods

    def log_likelihood(
        self,
        X: NDArray[np.float64],
        responses: NDArray[np.float64],
        fct_params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Compute the summed log-likelihood of all trials for each particle.

        Evaluates the Bernoulli log-likelihood for every (trial, particle) pair
        and sums across trials. Probabilities are clipped away from zero before
        taking logarithms to avoid ``-inf`` values.

        Parameters
        ----------
        X : ndarray of shape (n_trials, n_features)
            Stimuli presented across trials.
        responses : ndarray of shape (n_trials,)
            Observed binary responses (0 or 1) for each trial.
        fct_params : ndarray of shape (n_particles, n_parameters)
            Each row is a parameter vector for one particle.

        Returns
        -------
        ndarray of shape (n_particles,)
            Summed log-likelihood across all trials for each particle.
        """

        p = self.psychometric_function(X, fct_params)
        p = np.clip(p, 0 + 1e-100, 1)
        log_likelihoods = np.sum(
            np.log(np.where(responses, p, 1) * np.where(1 - responses, 1 - p, 1)),
            axis=1,
        )
        return log_likelihoods
