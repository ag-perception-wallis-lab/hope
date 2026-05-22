# %%
import numpy as np
from scipy import stats

from hope import HopeSampler
from hope.psychometric_functions import logistic_regression_lapses
from hope.psychometric_model import BinaryPsychometricModel

# %%
priors = {
    "lower_lapse": stats.beta(1, 30),
    "upper_lapse": stats.beta(30, 1),
    "bias": stats.norm(scale=np.sqrt(3)),
    "weights": stats.multivariate_normal(mean=np.zeros(2), cov=3 * np.eye(2)),
}
seed = 42
psychometric_model = BinaryPsychometricModel(
    psychometric_function=logistic_regression_lapses,
    priors=priors,
    bounds={"lower_lapse": (0, 0.2), "upper_lapse": (0.8, 1)},
    seed=seed,
)
rng = np.random.default_rng(seed)
stimulus_pool = rng.uniform(-2.5, 4, size=(500, 2))
sampler = HopeSampler(
    psychometric_model=psychometric_model,
    stimulus_pool=stimulus_pool,
    n_particles=1000,
    n_mh=5,
    seed=seed,
)
gt_parameters = np.array([0.01, 0.97, 0.5, 0.7, 1])
# %%
for i in range(5):
    stimulus = sampler.get_next_stimulus()
    print("Selected stimulus:", stimulus)
    response = np.random.binomial(
        1,
        psychometric_model.psychometric_function(stimulus, gt_parameters),
    )[0][0]
    print("Simulated response:", response)
    sampler.update_posterior(stimulus, response)
# %%
