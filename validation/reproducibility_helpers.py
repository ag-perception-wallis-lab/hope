"""
Helper functions ported from the original online-bayesian-optimization repository.

These are used only to validate that the new `hope` implementation reproduces results
from the original code — they are not part of the package's public API and are likely
not actively maintained going forward. There have been a few changes to make the
regression models compatible: In the original code, the bias was part of the regression
model's weights, here it is separate as the `bias` hyperparamter.
"""

import numpy as np
from scipy import stats
from typing import Callable, List

from jax import random
from numpyro.infer import MCMC, NUTS
from sklearn.neighbors import KDTree
from scipy.special import gamma
import numpyro
import numpyro.distributions as npdist
import jax.numpy as jnp
import jax.nn as jnn
import sys


def mcmc_approximation_numpyro(
    X: np.ndarray,
    responses: np.ndarray,
    model: Callable,
    prior: List[stats.rv_continuous],
    n_samples: int = 1000,
    n_chains: int = 1,
    n_warmup: int = 500,
    seed: int = None,
) -> np.ndarray:
    """Approximates the posterior distribution using Pyro based mcmc.

    Parameters
    ----------
    X : np.ndarray
        Stimuli
    responses : np.ndarray
        Responses for stimuli X.
    model : Callable[ [torch.FloatTensor, torch.IntTensor, List[stats.rv_continuous]], None ]
        Pyro MCMC Model
    prior : List[stats.rv_continuous]
        List of distributions which specify the prior.
    n_samples : int, optional
        Number of samples to be drawn by mcmc, by default 1000
    n_chains : int, optional
        Number of chains, by default 1
    n_warmup : int, optional
        Number of warm up steps (not included in n_samples), by default 500
    seed : int, optional
        random seed, by default None

    Returns
    -------
    np.ndarray
        Samples drawn from the approximated posterior distribution.
    """
    nuts_kernel = NUTS(model)
    # to change number of parallel chains just change num_chains to any number
    # use intitial_params to change the starting point of the chains (Doc: dict containing initial tensors in unconstrained space to initiate the markov chain. The leading dimension’s size must match that of num_chains. If not specified, parameter values will be sampled from the prior.)
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=n_chains,
    )
    if seed is None:
        seed = np.random.randint(0, 1000000000)
    rng_key = random.PRNGKey(seed)

    mcmc.run(rng_key, X=X.T, responses=responses.flatten(), prior=prior)
    return mcmc.get_samples()


def compute_posterior(
    X, responses, n_mcmc_samples_per_step, mcmc_log_reg_lapse_model, priors
):
    """Computes an MCMC posterior for the model, data and priors passed.

    Parameters
    ----------
    X : np.ndarray
        Input data.
    responses : np.ndarray
        Responses for the input data.
    n_mcmc_samples_per_step : int
        Number of MCMC samples per step.
    mcmc_log_reg_lapse_model : Callable
        The logistic regression with lapses model whose paramteres should be estimated.
    priors : List[stats.rv_continuous]
        The priors for the model parameters. Expects lower bound, upper bound, bias and
        weights.

    Returns
    -------
    np.ndarray
        The posterior samples.
    """
    mcmc_results = mcmc_approximation_numpyro(
        X=X,
        responses=responses,
        model=mcmc_log_reg_lapse_model,
        prior=priors,
        n_samples=n_mcmc_samples_per_step,
        n_warmup=1000,
    )

    posterior_samples = np.hstack(
        [
            mcmc_results["lower_bound"].reshape(-1, 1),
            mcmc_results["upper_bound"].reshape(-1, 1),
            mcmc_results["bias"].reshape(-1, 1),
            mcmc_results["theta"],
        ]
    )[::20]
    return posterior_samples


def nearest_neighbor_entropy(samples) -> float:
    """Approximates the entropy of a distribution using a nearest neighbor method.

    Parameters
    ----------
    samples : np.ndarray
        Samples drawn from the distribution.

    Returns
    -------
    float
        Approximated entropy of the distribution.
    """
    n_samples = samples.shape[0]
    tree = KDTree(samples)
    distances = tree.query(samples, k=2)[0][:, 1]
    p = samples.shape[1]
    eps = 1e-100
    return (
        p / n_samples * np.sum(np.log(distances + eps))
        + np.log(np.pi ** (p / 2) / gamma(p / 2 + 1))
        + np.euler_gamma
        + np.log(n_samples - 1)
    )


def mcmc_log_reg_lapse_model(
    X,
    responses,
    prior: List[stats.rv_continuous],
):
    """Pyro Model for logistic regression with lapses

    Parameters
    ----------
    x_torch : torch.FloatTensor
        Sampling positions of shape [n_samples, n_dim]
    response_torch : torch.IntTensor
         Responses to samples of shape [n_samples]
    prior : List[rv_continuous]
        List of length 4 specifying the prior distributions. prior[0] is the
        prior of the lower bound and has to be a Beta distribution. prior[1] is
        the prior of the upper bound and also has to be a Beta distribution.
        prior[2] is the prior of the bias. prior[3] is the prior of the weights
        and has to be be a multidimensional gaussian distribution with
        dim(distribution) = n_weights
    """
    lower_bound = numpyro.sample(
        "lower_bound", npdist.Beta(prior[0].args[0], prior[0].args[1])
    )
    upper_bound = numpyro.sample(
        "upper_bound", npdist.Beta(prior[1].args[0], prior[1].args[1])
    )
    lower_bound = jnp.where(lower_bound > 0.3, 0.3, lower_bound)
    upper_bound = jnp.where(upper_bound < 0.7, 0.7, upper_bound)

    # adaptation for new implementation of the bias
    bias = numpyro.sample(
        "bias", npdist.Normal(loc=prior[2].mean(), scale=jnp.sqrt(prior[2].var()))
    )
    theta = numpyro.sample(
        "theta",
        npdist.MultivariateNormal(
            loc=prior[3].mean,
            covariance_matrix=prior[3].cov,
        ),
    )
    response_probs = lower_bound + (upper_bound - lower_bound) * jnn.sigmoid(
        theta @ X + bias
    )
    response_probs = jnp.where(response_probs > 1, 1, response_probs)
    response_probs = jnp.where(response_probs < 0, 0, response_probs)

    with numpyro.plate("response_torch", responses.size):
        numpyro.sample(
            "obs",
            npdist.Bernoulli(probs=response_probs, validate_args=True),
            obs=responses,
        )


def in_interactive_mode() -> bool:
    """True if running in a Jupyter/IPython session or interactive Python shell.
    
    If not in interactive shell, plots should be saved instead of shown.
    """
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except ImportError:
        pass
    return hasattr(sys, "ps1")  # True in a plain interactive python shell