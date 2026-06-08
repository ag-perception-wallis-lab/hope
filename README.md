# High-dimensional Online Particle Estimation (HOPE) for psychophysical experiments

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/psihope.svg)](https://badge.fury.io/py/psihope)

TODO

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

TODO