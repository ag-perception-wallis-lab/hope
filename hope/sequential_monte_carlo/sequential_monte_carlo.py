from collections.abc import Callable
from typing import Tuple, Union

import numpy as np
from scipy import stats


def gaussian_regression_likelihood_importance_reweighting_wrapper(
    data: np.ndarray, locations: np.ndarray
) -> np.ndarray:
    """
    importance_reweighting uses the general interface of accepting locations and
    data and this function wraps gaussian_regression_likelihood to provide
    this interface.

    Parameters
    ----------
    locations : np.ndarray
        Locations is assumed to be an array of shape (n_particles,
    n_weights + 1) and the first n_weights columns represent the weights for the
    regression dimensions while the last column represents the noise variance.

    """
    return gaussian_regression_likelihood(
        data, weights=locations[:, 0:-1], variances=locations[:, -1]
    )


def gaussian_regression_likelihood(
    data: np.ndarray, weights: np.ndarray, variances: np.ndarray
) -> np.ndarray:
    """
    Provides the likelihoods for one or several datapoints given the regression
    coefficients and the noise variances.

    Parameters
    ----------
    data: tuple of nd.arrays
        X and y data of the regression. X is assumed to have dimensionality of
        (n_data, n_features). y can have a dimensinality of (n_data,) except
        when we have more than one data point. Then y needs to have shape
        (n_data, 1).
    weights : np.ndarray
        Regression coefficients sorted from highest polynomial to lowest (i.e.
        bias). Assumed shape is (n_particles, n_weights) (which can also be
        (n_weights,) when we have only one data point).
    variances : np.ndarray
        The noise variances for the Gaussian likelihoods to be computed.

    Returns
    -------
    np.ndarray
        The likelihoods at the different regression weight and noise variance
        locations.
    """
    X, y = data
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    if y.ndim == 1:
        y = np.expand_dims(y, axis=0)

    return np.prod(
        stats.norm.pdf(y, loc=X @ weights.T, scale=np.sqrt(variances)),
        axis=0,
    )


def importance_reweighting(
    locations: np.ndarray,
    weights: np.ndarray,
    data: Union[float, list, np.ndarray],
    likelihood_fn: Callable = lambda data, locations: np.prod(
        stats.norm.pdf(data, loc=locations, scale=1.0), axis=0
    ),
) -> np.ndarray:
    """
    This function updates the weights of the particles in an importance
    sampling manner according to the likelihood of the passed data and
    renormalizes them to one. Make sure that the data you pass is in a
    format that likelihood_fn works with.

    Parameters
    ----------
    locations : np.ndarray
        Locations of the particles.
    weights : np.ndarray
        Current weights of the particles.
    data : Union[float, list, np.ndarray]
        Data to use for the weight updating. Can be single data point or
        multiple ones.
    likelihood_fn : function
        A function that calculates the likelihood for one or many new data points.


    Returns
    -------
    np.ndarray
        The importance weights.

    """
    if np.isscalar(data):
        data = np.array([data])
        data = np.expand_dims(data, axis=1)
    updated_weights = weights * likelihood_fn(data, locations)
    updated_weights /= np.sum(updated_weights)
    return updated_weights


def importance_resampling(
    locations: np.ndarray, weights: np.ndarray, method="multinomial"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the importance resampling step. I.e. resamples new particles
    according to the importance weights and sets the weights of the new
    particles back to 1/n_particles.

    Parameters
    ----------
    locations : np.ndarray
        Particle locations
    weights : np.ndarray
        Particle weights

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        New particle locations and their uniform weights.
    """
    if locations.ndim == 1:
        locations = np.random.choice(
            locations, size=len(weights), replace=True, p=weights
        )
    else:
        locations = locations[
            np.random.choice(
                locations.shape[0], size=len(weights), replace=True, p=weights
            ),
            :,
        ]
    weights = np.ones_like(weights) / len(weights)
    return locations, weights


def n_eff(weights):
    """
    Calculates the effective number of particles based on the weights of the
    particles. If we do importance resampling, we should also take care of
    duplicate particles and if we additionally do MH we should take care of
    serial Markov dependencies in the particle locations.
    """
    return 1.0 / np.sum(np.square(weights))


def h(p):
    eps = 1e-100
    return -p * np.log(p + eps) - (1 - p) * np.log(1 - p + eps)


def mutual_information(probs):
    return h(1 / probs.shape[0] * np.sum(probs, axis=0)) - 1 / probs.shape[0] * np.sum(
        h(probs), axis=0
    )
