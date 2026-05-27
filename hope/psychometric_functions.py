import numpy as np
from scipy.special import expit


def logistic_regression_lapses(X, fct_params):
    if X.size == 1 and X.ndim <= 1:
        X = X.reshape((1, 1))
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    if fct_params.ndim == 1:
        fct_params = np.expand_dims(fct_params, axis=0)
    a = fct_params[:, 0][:, np.newaxis]
    k = fct_params[:, 1][:, np.newaxis]
    b = fct_params[:, 2][:, np.newaxis]
    s = expit(fct_params[:, 3:] @ X.T + b)
    p = a + (k - a) * s
    return p


def logistic_regression(X, fct_params):
    if X.size == 1 and X.ndim <= 1:
        X = X.reshape((1, 1))
    if X.ndim == 1:
        X = np.expand_dims(X, axis=0)
    if fct_params.ndim == 1:
        fct_params = np.expand_dims(fct_params, axis=0)
    b = fct_params[:, 0][:, np.newaxis]
    s = expit(fct_params[:, 1:] @ X.T + b)
    return s