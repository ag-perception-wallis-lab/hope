import numpy as np
from scipy.special import expit


def bounded_sigmoid(x, lower_bound, upper_bound):
    return lower_bound + (upper_bound - lower_bound) * expit(x)


def bounded_logit(x, lower_bound, upper_bound):
    return np.log(((x - lower_bound) / (upper_bound - x + 1e-50)) + 1e-50)