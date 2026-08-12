# High-dimensional Online Particle Estimation (HOPE) for psychophysical experiments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI - Version](https://img.shields.io/pypi/v/psihope)](https://pypi.org/project/psihope/)
[![DOI](https://zenodo.org/badge/1242238167.svg)](https://doi.org/10.5281/zenodo.20727919)

HOPE (High-dimensional Online Particle Estimation) is a Python package for adaptive psychophysics experiments. 

Given a parametric psychometric function and a discrete stimulus pool, it selects the next stimulus with maximum expected information gain. It uses a posterior approximation based on a combined particle filtering–MCMC approach. This is fast enough for real-time use within the inter-stimulus interval for feature spaces up to 50 dimensions, typically under one second on standard hardware and with a stimulus pool of 10,000 stimuli. The package includes a set of predefined psychometric functions, but users can also supply their own. It is designed to integrate with PsychoPy experiments.

A preprint of the paper describing and evaluating the method in simulations and a real experiment is available on [bioRxiv](https://doi.org/10.64898/2026.08.03.741989).

## Installation

You can install HOPE using pip:

```bash
pip install psihope
```

## Usage

```python
from hope import HopeSampler
from hope.psychometric_functions import logistic_regression
from hope.psychometric_model import BinaryPsychometricModel

# 1. Define your priors
priors = {
    "bias": stats.norm(scale=1),
    "weights": stats.multivariate_normal(mean=np.zeros(2), cov=np.eye(2)),
}

# 2. Define your model
psychometric_model = BinaryPsychometricModel(
    psychometric_function=logistic_regression, # choose from our library of psychometric functions or define your own
    priors=priors,
)

# 3. Initialize the sampler
# define your stimulus pool here as a list of stimulus configurations
stimulus_pool = ... 
sampler = HopeSampler(
    psychometric_model=psychometric_model,
    stimulus_pool=stimulus_pool,
    seed=seed,
)

# 4. Run the experiment
for trial in range(num_trials):
    stimulus = sampler.get_next_stimulus()

    # Collect response from participant using e.g. PsychoPy
    response = ...

    sampler.update_posterior(stimulus, response)
```

For more advanced usage and more explanations refer to the [examples](examples/) directory.

## Citation

If you use our tool, please cite our preprint:

```bibtex
@article{turon2026adaptive,
	author = {Turon, Rabea and Reining, Lars C. and Hummel, Philipp A. and Schmittwilken, Lynn and Lind, Christine and Yu, Angela J. and Rothkopf, Constantin A. and J{\"a}kel, Frank and Wallis, Thomas S. A.},
	title = {Adaptive experiments in high-dimensional feature spaces: A particle filtering approach},
	year = {2026},
	doi = {10.64898/2026.08.03.741989},
	url = {https://www.biorxiv.org/content/early/2026/08/07/2026.08.03.741989},
	eprint = {https://www.biorxiv.org/content/early/2026/08/07/2026.08.03.741989.full.pdf},
	journal = {bioRxiv}
}
```
