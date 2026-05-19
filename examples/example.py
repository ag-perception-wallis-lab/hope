# self.n_dims = n_dims
        # self.trans_prop = IntervalTransformIndependentGaussianProposer(
        #     [0, 1], np.array([0, 0.8]), np.array([0.2, 1])
        # )
        # prior = stats.multivariate_normal(
        #     mean=np.zeros(self.n_dims), variance=3, seed=seed
        # )
        # self.priors = {
        #     "lower_lapse": stats.beta(1, 30),
        #     "upper_lapse": stats.beta(1, 30),
        #     "weights": prior,
        #     "bias": stats.norm(scale=math.sqrt(3), seed=seed),
        # }
# %%
import numpy as np
from scipy import stats

from hope import HopeSampler
from hope.psychometric_functions import logistic_regression_lapses
from hope.psychometric_model import BinaryPsychometricModel

# %%
priors = {
            "lower_lapse": stats.beta(1, 30),
            "upper_lapse": stats.beta(1, 30),
            "bias": stats.norm(scale=np.sqrt(3)),
            "weights": stats.multivariate_normal(mean=np.zeros(2), cov=3 * np.eye(2)),
        }
psychometric_model = BinaryPsychometricModel(
    psychometric_function=logistic_regression_lapses,
    priors=priors,
    bounds={"lower_lapse": (0, 0.2), "upper_lapse": (0.8, 1)},
)
stimulus_pool = np.random.uniform(-2.5, 4, size=(500, 2))
sampler = HopeSampler(
    psychometric_model=psychometric_model,
    stimulus_pool=stimulus_pool,
    n_particles=3,
    n_mh=5,
)
# %%