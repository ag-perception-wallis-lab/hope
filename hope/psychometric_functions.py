import numpy as np
from scipy.special import expit


def logistic_regression_lapses(X, fct_params):
    if X.size == 1 and X.ndim <= 1:
        X = X.reshape((1, 1))
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    a = fct_params[:, 0]
    k = fct_params[:, 1]
    s = expit(fct_params[:, 2:] @ X.T)
    p = a + (k - a) * s
    return p