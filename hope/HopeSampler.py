import logging
import pickle
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from hope.psychometric_model import PsychometricModel

from .sequential_monte_carlo import (
    WeightedParticles,
    adapt_proposal_width_factor,
    importance_reweighting,
    metropolis_hastings_step,
    metropolis_step,
    mutual_information,
    propose_independent_gaussian,
    rule_of_thumb_bandwidths,
)

__all__ = ["HopeSampler"]
logger = logging.getLogger(__name__)


class HopeSampler:
    """Sampler for adaptive psychometric estimation.

    Maintains a particle approximation of the posterior over psychometric
    function parameters and selects stimuli that maximise expected information gain
    (i.e. minimise expected posterior entropy). After each trial the posterior
    is updated via importance reweighting, stratified resampling, and a
    configurable number of Metropolis (-Hastings) steps.

    Parameters
    ----------
    psychometric_model : PsychometricModel
        Model defining the psychometric function, priors, and (log-)likelihood.
    stimulus_pool : ndarray of shape (n_stimuli, n_features)
        Full set of candidate stimuli from which the next stimulus is selected.
    n_particles : int
        Number of particles used to represent the posterior.
    n_mh : int
        Number of Metropolis (-Hastings) steps performed after each posterior
        update.
    replace_after_trials : int
        Number of consecutive trials drawn without replacement before the full
        stimulus pool is restored. Set to 1 (default) to always sample with
        replacement. Clamped to ``stimulus_pool.shape[0]`` if larger.
    seed : int or None
        Seed for the internal ``numpy.random.Generator``.

    Attributes
    ----------
    n_mh : int
        Number of Metropolis (-Hastings) steps performed after each posterior
        update.
    n_particles : int
        Number of particles used to represent the posterior.
    particles : WeightedParticles
        Current particle approximation of the posterior, initialised from the
        prior.
    psychometric_model : PsychometricModel
        The psychometric model provided at initialisation.
    sampled : list[ndarray]
        Stimuli presented so far, in trial order.
    stimulus_pool : ndarray of shape (n_stimuli, n_features)
        Full set of candidate stimuli, refills X during replacement.
    responses : list[float]
        Responses recorded so far, in trial order.
    replace_after_trials : int
        Number of consecutive trials drawn without replacement before the full
        stimulus pool is restored. Set to 1 (default) to always sample with
        replacement. Clamped to ``stimulus_pool.shape[0]`` if larger.
    rng : numpy.random.Generator
        Internal random number generator, seeded from the constructor argument.
    proposal_width_factor : float
        Multiplicative scaling factor applied to rule-of-thumb bandwidths,
        adapted dynamically to target a reasonable MH acceptance rate.
    X : ndarray of shape (n_stimuli_remaining, n_features)
        Current stimulus pool from which the next stimulus is selected.
    """

    sampled: list[NDArray[np.float64]]
    responses: list[float]

    def __init__(
        self,
        psychometric_model: PsychometricModel,
        stimulus_pool: NDArray[np.float64],
        n_particles: int,
        n_mh: int,
        replace_after_trials: int = 1,
        seed: Optional[int] = None,
    ):

        self.rng = np.random.default_rng(seed)
        self.psychometric_model = psychometric_model
        self.n_mh = n_mh
        self.stimulus_pool = stimulus_pool  # all stimuli
        self.X = stimulus_pool  # current stimulus pool; might change if we sample without replacement
        if replace_after_trials > self.stimulus_pool.shape[0]:
            self.replace_after_trials = self.stimulus_pool.shape[0]
            logger.warning(
                "The value for replace_after_trials is bigger than the "
                "stimulus_pool size. To avoid drawing from an empty "
                "stimulus pool, replace_after_trials was set to the "
                "stimulus_pool size."
            )

        else:
            self.replace_after_trials = replace_after_trials

        self.n_particles = n_particles
        self.particles = WeightedParticles(
            np.array(self.psychometric_model.sample_prior(n_particles))
        )
        self.sampled = []
        self.responses = []
        self.proposal_width_factor = 1.0

    def get_next_stimulus(self) -> NDArray[np.float64]:
        """Computes and returns the stimulus in the current stimulus pool that
        minimizes the expected entropy.

        If replace_after_trials is set to a value bigger than 1 and the current
        stimulus pool size is bigger than the original stimulus pool size minus
        replace_after_trials, the selected stimulus is drawn from the pool without
        replacement. Once the difference between the current stimulus pool size and
        the original one has reached replace_after_trials, all stimuli are put back in
        the pool. If replace_after_trials equals 1, stimuli are always drawn with
        replacement.

        Returns
        -------
        ndarray of shape (n_features,)
            Stimulus in the current stimulus pool, that minimizes the expected entropy.
        """
        if self.stimulus_pool.shape[0] - self.X.shape[0] >= self.replace_after_trials:
            self.X = self.stimulus_pool.copy()
        probs = self.psychometric_model.psychometric_function(
            self.X, self.particles.locations
        )
        next = np.argmax(mutual_information(probs))
        stimulus = self.X[next]
        self.X = np.delete(self.X, next, axis=0)
        return stimulus

    def update_stimulus_pool(self, new_stimulus_pool: NDArray[np.float64]) -> None:
        """Replace the stimulus pool with a new set of stimuli.

        Parameters
        ----------
        new_stimulus_pool : ndarray of shape (n_stimuli, n_features)
            New stimulus pool.
        """

        self.stimulus_pool = new_stimulus_pool
        if self.replace_after_trials > self.stimulus_pool.shape[0]:
            self.replace_after_trials = self.stimulus_pool.shape[0]
            logger.warning(
                "The value for replace_after_trials is bigger than the "
                "stimulus_pool size. To avoid drawing from an empty "
                "stimulus pool, replace_after_trials was set to the "
                "stimulus_pool size."
            )

    def update_posterior(self, stimulus, response) -> None:
        """Update the particle posterior given a new stimulus–response pair.

        Appends the trial to the history, then performs:

        1. Importance reweighting using the single-trial likelihood.
        2. Particle resampling.
        3. ``n_mh`` Metropolis (-Hastings) steps evaluating the full posterior.
           Proposal variances are set via rule-of-thumb bandwidths for the
           first five steps and held fixed thereafter.
        4. Dynamic adaptation of ``proposal_width_factor`` based on the
           acceptance probabilities observed in step 3.

        If ``psychometric_model.trans_prop`` is set, an interval-transform MH
        step is used; otherwise a standard Metropolis step with an independent
        Gaussian proposal is used.

        Parameters
        ----------
        stimulus : ndarray of shape (n_features,)
            The stimulus that was presented.
        response : float
            The observed response (e.g. 0 or 1 for binary models).
        """

        self.sampled.append(stimulus)
        self.responses.append(response)
        self.particles.weights = importance_reweighting(
            self.particles.locations,
            self.particles.weights,
            [stimulus, response],
            self.psychometric_model.likelihood,
        )
        self.particles.importance_resampling(method="stratified", rng=self.rng)

        def unnormalized_log_posterior(
            particle_locations: NDArray[np.float64],
        ) -> NDArray[np.float64]:
            log_prior = self.psychometric_model.log_prior(particle_locations)
            ll = self.psychometric_model.log_likelihood(
                np.array(self.sampled),
                np.array(self.responses),
                particle_locations,
            )
            return log_prior + ll

        j = 0
        acceptance_probs: list[float] = []
        # do some mcmc steps (rule to be implemented) TODO
        logger.debug("Starting MH steps")
        for j in range(self.n_mh):
            logger.debug(f"MH step {j + 1}/{self.n_mh}")
            if self.psychometric_model.trans_prop is not None:
                if j < 5:
                    proposal_vars = (
                        self.proposal_width_factor
                        * rule_of_thumb_bandwidths(
                            self.psychometric_model.trans_prop.transform(
                                self.particles.locations
                            ),
                            effective_n=self.particles.n_particles
                            - self.particles.duplicate_ratio
                            * self.particles.n_particles,
                        )
                    )
                    self.psychometric_model.trans_prop.proposal_vars = proposal_vars
                (
                    new_locations,
                    indices_of_updated_locations,
                    acceptance_prob,
                ) = metropolis_hastings_step(
                    self.particles.locations,
                    unnormalized_log_posterior_fn=unnormalized_log_posterior,
                    trans_proposer=self.psychometric_model.trans_prop,
                    rng=self.rng,
                )
                logger.debug("duplicate ratio ", self.particles.duplicate_ratio)
                logger.debug("proposal_width_factor:", self.proposal_width_factor)
                logger.debug("proposal_vars:", proposal_vars)
                logger.debug("acceptance_probs:", acceptance_prob)
            else:
                if j < 5:
                    proposal_vars = (
                        self.proposal_width_factor
                        * rule_of_thumb_bandwidths(
                            self.particles.locations,
                            effective_n=self.particles.n_particles
                            - self.particles.duplicate_ratio
                            * self.particles.n_particles,
                        )
                    )
                (
                    new_locations,
                    indices_of_updated_locations,
                    acceptance_prob,
                ) = metropolis_step(
                    self.particles.locations,
                    unnormalized_log_posterior_fn=unnormalized_log_posterior,
                    proposal_fn=propose_independent_gaussian,
                    proposal_vars=proposal_vars,
                    rng=self.rng,
                )
                logger.debug("duplicate ratio ", self.particles.duplicate_ratio)
                logger.debug("proposal_width_factor:", self.proposal_width_factor)
                logger.debug("proposal_vars:", proposal_vars)
                logger.debug("acceptance_probs:", acceptance_prob)
            acceptance_probs.append(acceptance_prob)
            self.particles.update_locations(new_locations, indices_of_updated_locations)
        # dynamically adapt proposal width
        self.proposal_width_factor = adapt_proposal_width_factor(
            self.proposal_width_factor, acceptance_probs
        )

    def save(self, path: str) -> None:
        """Save the sampler to disk using pickle.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "HopeSampler":
        """Load an existing sampler previously saved with ``save``.

        Parameters
        ----------
        path : str
            Path to the pickle file.

        Returns
        -------
        HopeSampler
            The restored sampler instance.
        """
        with open(path, "rb") as f:
            return pickle.load(f)
