import numpy as np

from .sequential_monte_carlo import WeightedParticles


class PsychometricModel(ABC):
    def __init__(
        self, prior_params
    ):  # TODO: talk to Swantje to check how to pass distributions the best
        self.prior_params = prior_params

    @abstractmethod
    def likelihood(self, X, responses, fct_params):
        pass

    @abstractmethod
    def log_likelihood(self, X, reponses, fct_params):
        pass

    def sample_prior(self, n_samples) -> np.ndarray:
        pass


class LogisticRegressionWithLapses(PsychometricModel):
    def __init__(
        self, n_dims, priors
    ):  # TODO: talk to Swantje to check how to pass distributions the best
        super().__init__()
        self.n_dims = n_dims
        # TODO: do we want the priors here or params for the priors instead?
        #       also: checks to see if the order fits what we expect (lapses have
        #       other priors than other parameters). If we decide to use params here
        #       we have to adapt the base class.
        self.trans_prop = trans_prop  # TODO: look up what the trans_prop should be
        self.priors = priors

    @staticmethod
    def likelihood(X, responses, fct_params):
        if X.size == 1 and X.ndim <= 1:
            X = X.reshape((1, 1))
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        a = fct_params[:, 0]
        k = fct_params[:, 1]
        p = a + (k - a) * expit(X @ fct_params[:, 2:].T)
        p[p > 1] = 1
        p[p < 0] = 0 + 1e-100
        likelihoods = np.prod(
            (np.power(p, responses) * np.power(1 - p, 1 - responses)), axis=0
        )
        likelihoods[likelihoods > 1] = 1
        likelihoods[likelihoods < 0] = 0
        return likelihoods

    @staticmethod
    def log_likelihood(X, responses, t):
        a = t[:, 0].reshape(-1, 1)
        k = t[:, 1].reshape(-1, 1)
        s = expit(t[:, 2:] @ X.T)
        p = a + (k - a) * s
        p = np.clip(p, 0 + 1e-100, 1)
        res = np.sum(
            np.log(np.where(responses, p, 1) * np.where(1 - responses, 1 - p, 1)),
            axis=1,
        )
        return res

    # @staticmethod
    # def likelihood_importance_reweighting_wrapper(data, locations):
    #     x, responses = data
    #     x = np.hstack([1, x])
    #     return LogisticLapseRegressionLikelihood.likelihood(x, responses, locations)

    def sample_prior(self, n_samples) -> np.ndarray:
        def initialize_posteriors(
            priors, trans_prop, n_particles
        ):  # some function that samples the first particles from priors
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
