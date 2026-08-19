# %%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from time import time
import platform

from hope import HopeSampler
from hope.psychometric_functions import logistic_regression_lapses
from hope.psychometric_model import BinaryPsychometricModel

seed = 7
np.random.seed(seed)

n_trials = 1000
n_runs = 3
dims = [2,4,8,15,30,50]

times_select_stimulus = np.zeros((len(dims), n_runs, n_trials))
times_posterior_update = np.zeros((len(dims), n_runs, n_trials))

for i, n_dims in enumerate(dims):
    for j in range(n_runs):
        print("i and j:", i, j)
        run_seed = seed + i + j
        rng = np.random.default_rng(run_seed)
        stimulus_pool = rng.uniform(-5, 5, size=(10000, n_dims))

        priors = {
            "lower_lapse": stats.beta(1, 30),
            "upper_lapse": stats.beta(30, 1),
            "bias": stats.norm(scale=1),
            "weights": stats.multivariate_normal(mean=np.zeros(n_dims), cov=1),
        }
        psychometric_model = BinaryPsychometricModel(
            psychometric_function=logistic_regression_lapses,
            priors=priors,
            bounds={"lower_lapse": (0, 0.2), "upper_lapse": (0.8, 1)},
            seed=run_seed,
        )

        gt_parameters = np.hstack(
            (
                priors["lower_lapse"].rvs(),
                priors["upper_lapse"].rvs(),
                priors["bias"].rvs(),
                priors["weights"].rvs()
            )
        )

        hope_sampler = HopeSampler(
            psychometric_model=psychometric_model,
            stimulus_pool=stimulus_pool,
            n_particles=1000,
            n_mh=5,
            seed=run_seed,
        )

        # run adaptive trials
        for k in tqdm(range(n_trials), desc=f"Running hope simulations for dimensionality {n_dims}"):
            time_before_select = time()
            stimulus_hope = hope_sampler.get_next_stimulus()
            # print(stimulus_hope.shape)
            # print(stimulus_hope)
            times_select_stimulus[i,j,k] = (time()-time_before_select)

            response_hope = np.random.binomial(
                1,
                psychometric_model.psychometric_function(stimulus_hope, gt_parameters),
            )[0][0]
            time_before_update = time()

            hope_sampler.update_posterior(stimulus_hope, response_hope)
            times_posterior_update[i,j,k] = time()-time_before_update


        RESULTS_DIR = Path(__file__).parent / "results"
        np.savez(
            RESULTS_DIR / f"times_{n_dims}D_{n_trials}-trials.npz",
            times_select_stimulus=times_select_stimulus,
            times_posterior_update=times_posterior_update,
        )
# %%
plt.figure()
system_info = platform.platform()
for i in range(times_select_stimulus.shape[0]):
    plt.plot(times_select_stimulus[i].mean(axis=0), color="orange", label="stimulus selection time")
    plt.plot(times_posterior_update[i].mean(axis=0), color="blue", label="posterior update time")
    isi = times_posterior_update[i].mean(axis=0) + times_select_stimulus[i].mean(axis=0)
    plt.plot(isi, color="green", label="inter stimulus interval")

plt.title(f"Platform: {system_info}")
plt.legend()
plt.show()

# %%
