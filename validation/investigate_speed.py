# %%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm import tqdm
from pathlib import Path
from time import time
import platform

from hope import HopeSampler
from hope.psychometric_functions import logistic_regression_lapses
from hope.psychometric_model import BinaryPsychometricModel
from reproducibility_helpers import in_interactive_mode

RESULTS_DIR = Path(__file__).parent / "results"

seed = 7
# %%
np.random.seed(seed)

n_trials = 1000
n_runs = 10
dims = [2,4,8,15,30,50]
# %%
times_select_stimulus = np.zeros((len(dims), n_runs, n_trials))
times_posterior_update = np.zeros((len(dims), n_runs, n_trials))

for i, n_dims in enumerate(dims):  
    out_path = RESULTS_DIR / f"times_{n_dims}D_{n_trials}-trials_{n_runs}-runs.npz"
    if out_path.exists():
        print(f"Skipping {n_dims}D, already computed.")
        continue

    for j in range(n_runs):
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
            times_select_stimulus[i,j,k] = (time()-time_before_select)
            response_hope = np.random.binomial(
                1,
                psychometric_model.psychometric_function(stimulus_hope, gt_parameters),
            )[0][0]

            time_before_update = time()
            hope_sampler.update_posterior(stimulus_hope, response_hope)
            times_posterior_update[i,j,k] = time()-time_before_update

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS_DIR / f"times_{n_dims}D_{n_trials}-trials_{n_runs}-runs.npz",
        times_select_stimulus=times_select_stimulus[i],
        times_posterior_update=times_posterior_update[i],
    )
# %%

times_select_stimulus = {}
times_posterior_update = {}

for n_dims in dims:
    filepath = RESULTS_DIR / f"times_{n_dims}D_{n_trials}-trials_{n_runs}-runs.npz"
    if not filepath.exists():
        print(f"Warning: missing results for {n_dims}D at {filepath}")
        continue
    data = np.load(filepath)
    times_select_stimulus[n_dims] = data["times_select_stimulus"]
    times_posterior_update[n_dims] = data["times_posterior_update"]

print("Loaded dimensionalities:", list(times_select_stimulus.keys()))

plt.figure(figsize=(10, 6))
system_info = platform.platform()

dims_list = list(times_select_stimulus.keys())
colors = plt.cm.cividis(np.linspace(0, 0.9, len(dims_list))) 
colors = plt.cm.tab10(np.linspace(0, 1.0, len(dims_list))) 
 
linestyles = {
    "selection": "-",
    "update": "--",
    "isi": ":",
}

for color, n_dims in zip(colors, dims_list):
    select = times_select_stimulus[n_dims].mean(axis=0)
    update = times_posterior_update[n_dims].mean(axis=0)
    isi = select + update

    plt.plot(select, color=color, linestyle=linestyles["selection"], alpha=0.7)
    plt.plot(update, color=color, linestyle=linestyles["update"], alpha=0.7)
    plt.plot(isi, color=color, linestyle=linestyles["isi"], alpha=0.7)

# Legend part 1: color -> dimensionality
dim_handles = [
    mlines.Line2D([], [], color=color, label=f"{n_dims}D")
    for color, n_dims in zip(colors, dims_list)
]
# Legend part 2: linestyle -> metric
style_handles = [
    mlines.Line2D([], [], color="black", linestyle=ls, label=name)
    for name, ls in [
        ("Stimulus selection", linestyles["selection"]),
        ("Posterior update", linestyles["update"]),
        ("Inter-stimulus interval", linestyles["isi"]),
    ]
]

first_legend = plt.legend(
    handles=dim_handles, title="Dimensionality",
    bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.
)
plt.gca().add_artist(first_legend)
plt.legend(
    handles=style_handles, title="Metric",
    bbox_to_anchor=(1.02, 0.5), loc="upper left", borderaxespad=0.
)

plt.xlabel("Trial")
plt.ylabel("Time (s)")
plt.title(f"Platform: {system_info}")
plt.tight_layout()

if in_interactive_mode():
    plt.show()
else:
    filename = f"timing_plot_{n_runs}-runs.png"
    plt.savefig(RESULTS_DIR / filename, dpi=150, bbox_inches="tight")
    print(f"Saved plot to '{RESULTS_DIR / filename}'")


# %%
