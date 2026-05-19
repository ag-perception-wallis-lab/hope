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


# TODO: Think about loading previous HopeSampler
class HopeSampler:
    def __init__(
        self,
        psychometric_model: PsychometricModel,
        stimulus_pool: np.ndarray,  # TODO stimulus pool --> should we add other option with range with min and max?
        n_particles: int,
        n_mh: int,
        proposal_dist=None,  # TODO document default proposal distribution
        replace_after_trials: int = 1,
        seed=None,  # TODO implement seeding
    ):
        # TODO docstring
        self.rng = np.random.default_rng(seed)
        self.psychometric_model = psychometric_model
        self.proposal_dist = proposal_dist
        self.n_mh = n_mh
        self.stimulus_pool = stimulus_pool  # all stimuli
        self.X = stimulus_pool  # current stimulus pool; might change if we sample without replacement
        self.replace_after_trials = replace_after_trials
        self.n_particles = n_particles
        self.particles = WeightedParticles(
            np.array(self.psychometric_model.sample_prior(n_particles))
        )
        self.sampled = []
        self.responses = []

    def get_next_stimulus(self):
        # TODO add documentation about sampling with replacement and without replacement and how the stimulus pool is handled in both cases
        if (
            self.stimulus_pool.shape[0] - self.X.shape[0] >= self.replace_after_trials
        ):  # TODO catch stimulus pool empty error
            self.X = self.stimulus_pool.copy()
        probs = self.psychometric_model.psychometric_function(self.X, self.particles)
        next = np.argmax(mutual_information(probs))
        stimulus = self.X[next]
        self.X = np.delete(self.X, next, axis=0)
        return stimulus

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
                np.array(self.sampled), np.array(self.responses).T, particle_locations
            )
            return log_prior + ll

        j = 0
        acceptance_probs = []
        proposal_width_factor = 1.0
        # do some mcmc steps (rule to be implemented) TODO
        while j < self.n_mh:
            if self.seed is not None:
                self.seed = self.seed + j
            if self.psychometric_model.trans_prop is not None:
                if j < 5:
                    proposal_vars = proposal_width_factor * rule_of_thumb_bandwidths(
                        self.psychometric_model.trans_prop.transform(
                            self.particles.locations
                        ),
                        effective_n=self.particles.n_particles
                        - self.particles.duplicate_ratio * self.particles.n_particles,
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
            else:
                if j < 5:
                    proposal_vars = proposal_width_factor * rule_of_thumb_bandwidths(
                        self.particles.locations,
                        effective_n=self.particles.n_particles
                        - self.particles.duplicate_ratio * self.particles.n_particles,
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
            # todo when are we doing mh or h steps, document properly
            acceptance_probs.append(acceptance_prob)
            self.particles.update_locations(new_locations, indices_of_updated_locations)
            j += 1
        # dynamically adapt proposal width
        proposal_width_factor = adapt_proposal_width_factor(
            proposal_width_factor, acceptance_probs
        )