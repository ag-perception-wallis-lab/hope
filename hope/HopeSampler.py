import logging
import pickle

import numpy as np

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
logger = logging.getLogger(__file__)


class HopeSampler:
    def __init__(
        self,
        psychometric_model: PsychometricModel,
        stimulus_pool: np.ndarray,
        n_particles: int,
        n_mh: int,
        proposal_dist=None,  # TODO document default proposal distribution
        replace_after_trials: int = 1,
        seed=None,
    ):
        # TODO docstring
        self.rng = np.random.default_rng(seed)
        self.psychometric_model = psychometric_model
        self.proposal_dist = proposal_dist
        self.n_mh = n_mh
        self.stimulus_pool = stimulus_pool  # all stimuli
        self.X = stimulus_pool  # current stimulus pool; might change if we sample without replacement
        if replace_after_trials > self.stimulus_pool.shape[0]:
            self.replace_after_trials = self.stimulus_pool.shape[0]
            warning_str = (
                "The value for replace_after_trials is bigger than the"
                "stimulus_pool size. To avoid drawing from an empty "
                "stimulus pool, replace_after_trials was set to the "
                "stimulus_pool size."
            )
            logger.warning(warning_str)
        else:
            self.replace_after_trials = replace_after_trials

        self.n_particles = n_particles
        self.particles = WeightedParticles(
            np.array(self.psychometric_model.sample_prior(n_particles))
        )
        self.sampled = []
        self.responses = []
        self.proposal_width_factor = 1.0


    def get_next_stimulus(self):
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
        np.ndarray
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

    def update_stimulus_pool(self, new_stimulus_pool):
        self.stimulus_pool = new_stimulus_pool
        if self.replace_after_trials > self.stimulus_pool.shape[0]:
            self.replace_after_trials = self.stimulus_pool.shape[0]
            warning_str = (
                "The value for replace_after_trials is bigger than the"
                "stimulus_pool size. To avoid drawing from an empty "
                "stimulus pool, replace_after_trials was set to the "
                "stimulus_pool size."
            )
            logger.warning(warning_str)

    def update_posterior(self, stimulus, response):
        self.sampled.append(stimulus)
        self.responses.append(response)
        self.particles.weights = importance_reweighting(
            self.particles.locations,
            self.particles.weights,
            [stimulus, response],
            self.psychometric_model.likelihood,
        )
        self.particles.importance_resampling(method="stratified", rng=self.rng)
        def unnormalized_log_posterior(particle_locations):
            log_prior = self.psychometric_model.log_prior(particle_locations)
            ll = self.psychometric_model.log_likelihood(
                np.array(self.sampled),
                np.array(self.responses),
                particle_locations,
            )
            return log_prior + ll

        j = 0
        acceptance_probs = []
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
            # todo when are we doing mh or h steps, document properly
            acceptance_probs.append(acceptance_prob)
            self.particles.update_locations(new_locations, indices_of_updated_locations)
        # dynamically adapt proposal width
        self.proposal_width_factor = adapt_proposal_width_factor(
            self.proposal_width_factor, acceptance_probs
        )

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "HopeSampler":
        with open(path, "rb") as f:
            return pickle.load(f)
