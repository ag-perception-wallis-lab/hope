__all__ = ["HopeSampler"]


class HopeSampler():
    def __init__(
        self,
        psychometric_model,
        stimulus_pool, # stimulus pool --> should we add other option with range with min and max?
        n_particles,
        n_mh,
	    proposal_dist,
        replace_after_trials=1, # filling up of stimulus pool after this amount of trials
        seed=None,
    ):
        # TODO: Think about loading previous HopeSampler
        self.psychometric_model = psychometric_model
        self.proposal_dist = proposal_dist
        self.n_mh = n_mh
        self.stimulus_pool = stimulus_pool # all stimuli
        self.X = stimulus_pool # current stimulus pool; might change if we sample without replacement
        self.replace_after_trials = replace_after_trials


        def initialize_posteriors(priors, trans_prop, n_particles):  # some function that samples the first particles from priors
            rng = np.random.default_rng(seed)
            # sample from prior to initialize particles
            prior_samples = []
            only_one = True
            for i, prior in enumerate(priors):
                if i != len(priors) - 1:
                    new_samples = prior.rvs(size=n_particles, random_state=seed)
                    if trans_prop:
                        new_samples = new_samples.clip(
                            trans_prop.lower_bounds[i], trans_prop.upper_bounds[i]
                        )
                    prior_samples = prior_samples + [new_samples.tolist()]
                    only_one = False
                else:
                    if only_one:
                        prior_samples = prior.rvs(size=n_particles, random_state=seed)
                        break
                    prior_samples = np.array(prior_samples).T
                    prior_samples = np.hstack(
                        [prior_samples, prior.rvs(size=n_particles, random_state=seed)]
                    )
            # if initial_particles is None:
            particles = WeightedParticles(prior_samples)
            logging.info("Initialized particles from priors.")
            return particles

        # TODO: initial particles should not be here anymore now?
        self.particles = initialize_posteriors(
            psychometric_model.priors,
            psychometric_model.trans_prop,
            n_particles)
        self.sampled = []
        self.responses = []

    def get_next_stimulus(self):
        """Copied and deleted code from data_sampling.py PsiSampler

        Returns: next stimulus 
        """
        # if self.X.shape[0] < self.min_n_samples:
        #     self.generate_samples(rng)
        # if not rng:
        #     rng = np.random.default_rng()
        # if self.particles is not None:
        probs = self.likelihood(self.X, self.particles)
        next = np.argmax(mutual_information(probs))
        # else:
        #     next = rng.integers(0, self.X.shape[0])
        x = self.X[next]
        # response = self.responder.get_response(x)
        
        self.X = np.delete(self.X, next, axis=0)
        return x
    
    def update_posterior(self, stimulus, response):
        self.sampled.append(stimulus)
        self.responses.append(response)

        # now use the new information to update the posterior
        self.particles.weights = importance_reweighting(
            self.particles.locations,
            self.particles.weights,
            [_x[1:], _res],
            Likelihood.likelihood_importance_reweighting_wrapper,
        )
        self.particles.importance_resampling(method="stratified", rng=rng)
        particle_duplicate_ratio = self.particles.duplicate_ratio
        logging.debug("_x:", _x)
        logging.debug("_res:", _res)
        logging.debug(f"Effective N of self.particles: {n_eff(self.particles.weights)}")
        logging.debug(f"Duplicate Ratio: {particle_duplicate_ratio}")
        logging.debug("Starting Metropolis")

        def unnormalized_log_posterior(samples):
            log_prior = 0
            h = 0
            for prior in priors[:-1]:
                log_prior += prior.logpdf(samples[:, h])
                h += 1
            log_prior += priors[-1].logpdf(samples[:, h:])
            ll = Likelihood.log_likelihood(np.array(X), np.array(res).T, samples)
            return log_prior + ll

        j = 0
        acceptance_probs = []

        # do some mcmc steps (rule to be implemented)
        while j < self.n_mh:
            if store_run_config:
                save()
            if seed is not None:
                seed = seed + j
            if self.trans_prop is not None:
                if j < 5:
                    proposal_vars = proposal_width_factor * rule_of_thumb_bandwidths(
                        trans_prop.transform(self.particles.locations),
                        effective_n=self.particles.n_particles
                        - self.particles.duplicate_ratio * self.particles.n_particles,
                    )
                    trans_prop.proposal_vars = proposal_vars
                (
                    new_locations,
                    indices_of_updated_locations,
                    acceptance_prob,
                ) = metropolis_hastings_step(
                    self.particles.locations,
                    unnormalized_log_posterior_fn=unnormalized_log_posterior,
                    trans_proposer=trans_prop,
                    rng=rng,
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
                    rng=rng,
                )
            acceptance_probs.append(acceptance_prob)
            self.particles.update_locations(new_locations, indices_of_updated_locations)
            j += 1
            logging.debug(f"MH step {j}")
            logging.debug(f"Acceptance Prob: {acceptance_prob}")
            logging.debug(f"Duplicate Ratio: {self.particles.duplicate_ratio}")

        logging.info(f"{j} metropolis (hastings) steps were made.")
        if store_run_config:
            save()

        # dynamically adapt proposal width
        proposal_width_factor = adapt_proposal_width_factor(
            proposal_width_factor, acceptance_probs
        )
        logging.debug(f"New proposal width factor: {proposal_width_factor}")
        if data_sampler.responder.finished:
            if not store_particle_positions:
                store_particle_positions = True
                save()
            break

