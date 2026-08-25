# %%
import numpy as np
from scipy import stats
import sklearn.linear_model as lm
import scipy.io as io
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from hope import HopeSampler
from hope.psychometric_functions import logistic_regression_lapses
from hope.psychometric_model import BinaryPsychometricModel
from reproducibility_helpers import (
    compute_posterior,
    nearest_neighbor_entropy,
    mcmc_log_reg_lapse_model,
)

DATA_DIR = Path(__file__).parent / "aam_data"
seed = 42
np.random.seed(seed)

n_hope = 30
n_uniform = 90

priors = {
    "lower_lapse": stats.beta(1, 30),
    "upper_lapse": stats.beta(30, 1),
    "bias": stats.norm(scale=1),  # scale was np.sqrt(3), but why that specifically?
    "weights": stats.multivariate_normal(mean=np.zeros(15), cov=1),
}

psychometric_model = BinaryPsychometricModel(
    psychometric_function=logistic_regression_lapses,
    priors=priors,
    bounds={"lower_lapse": (0, 0.2), "upper_lapse": (0.8, 1)},
    seed=seed,
)

# create observer on cfd labels
cfd_faces = np.loadtxt(DATA_DIR / "cfd_aam_BW_15PCs.csv")
column_of_ones = np.ones(cfd_faces.shape[0])
cfd_faces = np.insert(cfd_faces, 0, column_of_ones, axis=1)
cfd_labels = io.loadmat(DATA_DIR / "CFDn_demo.mat")["target"][:, 0]
data_train, data_test, labels_train, labels_test = train_test_split(
    cfd_faces, cfd_labels, test_size=0.2
)

gt_weights = (
    lm.LogisticRegression(fit_intercept=False).fit(data_train, labels_train).coef_
)
gt_parameters = np.hstack((np.zeros((1, 1)), np.ones((1, 1)), gt_weights))


stimulus_lookup = pd.read_csv(
    DATA_DIR / "cfd_aam_BW_15PCs_pc_sample_lookup.csv", index_col=0
)
stimulus_pool = stimulus_lookup.values

hope_sampler = HopeSampler(
    psychometric_model=psychometric_model,
    stimulus_pool=stimulus_pool,
    n_particles=1000,
    n_mh=5,
    seed=seed,
)
uniform_sampler = HopeSampler(
    psychometric_model=psychometric_model,
    stimulus_pool=stimulus_pool,
    n_particles=1000,
    n_mh=5,
    seed=seed,
)

entropy_hope = []
entropy_uniform = []

monitored_trials = [
    1,
    3,
    5,
    9,
    14,
    19,
    28,
    42,
    63,
    85,
    110,
    140,
    170,
    200,
    250,
    320,
    410,
    540,
    710,
    930,
    1000,
    1250,
    1500,
    1750,
    2000,
    2250,
    2500,
    2750,
    3000,
    3500,
    4000,
    4500,
    5000,
    5500,
    6000,
    6500,
    7000,
    7500,
    8000,
    8500,
    9000,
]
n_mcmc_samples_per_step = 20000
priors_list = list(priors.values())

# run adaptive
for i in tqdm(range(n_hope), desc="Running hope simulations"):
    stimulus_hope = hope_sampler.get_next_stimulus()
    response_hope = np.random.binomial(
        1,
        psychometric_model.psychometric_function(stimulus_hope, gt_parameters),
    )[0][0]
    hope_sampler.update_posterior(stimulus_hope, response_hope)

    if i in monitored_trials:
        posterior_hope = compute_posterior(
            np.array(hope_sampler.sampled),
            np.array(hope_sampler.responses),
            n_mcmc_samples_per_step=n_mcmc_samples_per_step,
            mcmc_log_reg_lapse_model=mcmc_log_reg_lapse_model,
            priors=priors_list,
        )
        entropy_hope.append(nearest_neighbor_entropy(posterior_hope))


# run uniform
for i in tqdm(range(n_uniform), desc="Running uniform simulations"):
    stimulus_uniform = stimulus_pool[np.random.choice(stimulus_pool.shape[0])]
    response_uniform = np.random.binomial(
        1,
        psychometric_model.psychometric_function(stimulus_uniform, gt_parameters),
    )[0][0]
    uniform_sampler.update_posterior(stimulus_uniform, response_uniform)
    if i in monitored_trials:
        posterior_uniform = compute_posterior(
            np.array(uniform_sampler.sampled),
            np.array(uniform_sampler.responses),
            n_mcmc_samples_per_step=n_mcmc_samples_per_step,
            mcmc_log_reg_lapse_model=mcmc_log_reg_lapse_model,
            priors=priors_list,
        )
        entropy_uniform.append(nearest_neighbor_entropy(posterior_uniform))

RESULTS_DIR = Path(__file__).parent / "results"
np.savez(
    RESULTS_DIR / f"entropy_face_simulations_{n_hope}_{n_uniform}.npz",
    monitored_trials=monitored_trials,
    entropy_hope=entropy_hope,
    entropy_uniform=entropy_uniform,
)

plt.figure()
plt.plot(monitored_trials[: len(entropy_uniform)], entropy_uniform, color="blue")
plt.plot(monitored_trials[: len(entropy_hope)], entropy_hope, color="orange")
plt.show()
# %%
