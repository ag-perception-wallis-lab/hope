from collections.abc import Callable
from typing import Tuple, Union

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def importance_reweighting(
    locations: NDArray[np.float64],
    weights: NDArray[np.float64],
    data: Union[float, list, NDArray[np.float64]],
    likelihood_fn: Callable = lambda data, locations: np.prod(
        stats.norm.pdf(data, loc=locations, scale=1.0), axis=0
    ),
) -> NDArray[np.float64]:
    """Update particle weights by importance reweighting and renormalise.

    Multiplies the current weights by the likelihood of ``data`` under each
    particle, then normalises so that the weights sum to one.

    Parameters
    ----------
    locations : ndarray of shape (n_particles, n_dimensions)
        Current particle locations.
    weights : ndarray of shape (n_particles,)
        Current particle weights.
    data : float, list, or ndarray
        Data to use for the weight updating. Can be single data point or
        multiple ones. Unpacked as ``(stimulus, response)``.
    likelihood_fn : Callable
        Function with signature ``f(stimulus, response, locations) ->
        ndarray of shape (n_particles,)`` returning the likelihood of the
        data for each particle. Defaults to a standard normal likelihood.

    Returns
    -------
    ndarray of shape (n_particles,)
        Updated and normalized importance weights.
    """

    if np.isscalar(data):
        data = np.array([data])
        data = np.expand_dims(data, axis=1)
    x, response = data
    updated_weights = weights * likelihood_fn(x, response, locations)
    updated_weights /= np.sum(updated_weights)
    return updated_weights


def n_eff(weights: NDArray[np.float64]) -> float:
    """
    Calculates the effective number of particles based on the weights of the
    particles. If we do importance resampling, we should also take care of
    duplicate particles and if we additionally do MH we should take care of
    serial Markov dependencies in the particle locations.

    Parameters
    ----------
    weights : ndarray of shape (n_particles,)
        Normalized particle weights (must sum to 1).

    Returns
    -------
    float
        Effective sample size in the range ``[1, n_particles]``.
    """

    return 1.0 / np.sum(np.square(weights))


def binary_entropy(p: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute the binary entropy ``H(p) = -p log p - (1-p) log(1-p)`` element-wise.

    A small epsilon is added inside the logarithms to avoid ``log(0)``.

    Parameters
    ----------
    p : ndarray
        Bernoulli probabilities. Values should lie in ``[0, 1]``.

    Returns
    -------
    ndarray
        Binary entropy of ``p``, same shape as input.
    """
    eps = 1e-100
    return -p * np.log(p + eps) - (1 - p) * np.log(1 - p + eps)


def mutual_information(probs: NDArray[np.float64]) -> NDArray[np.float64]:
    """Estimate the mutual information between stimuli and binary responses under the current posterior.

    Computes ``H(E[p]) - E[H(p)]``, where the expectation is taken over
    particles (i.e. the current posterior approximation). A higher value
    indicates that presenting the stimulus is expected to reduce posterior
    entropy more.

    Parameters
    ----------
    probs : ndarray of shape (n_particles, n_stimuli)
        Predicted response probability for each particle–stimulus combination.

    Returns
    -------
    ndarray of shape (n_stimuli,)
        Estimated mutual information for each candidate stimulus.
    """
    return binary_entropy(1 / probs.shape[0] * np.sum(probs, axis=0)) - 1 / probs.shape[
        0
    ] * np.sum(binary_entropy(probs), axis=0)
