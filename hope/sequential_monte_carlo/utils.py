import numpy as np
from numpy.typing import NDArray
from scipy.special import expit


def bounded_sigmoid(
    x: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Map values from the real line into per-element open intervals ``(lower_bounds, upper_bounds)``.

    Applies a linearly scaled logistic (sigmoid) function element-wise:
    ``lower_bounds + (upper_bounds - lower_bounds) * sigmoid(x)``.
    This is the inverse of ``bounded_logit``.

    Parameters
    ----------
    x : ndarray of shape (..., n_transformed_dimensions)
        Values in ``(-inf, +inf)``.
    lower_bounds : ndarray of shape (n_transformed_dimensions,)
        Per-element lower bounds, broadcastable against ``x``.
    upper_bounds : ndarray of shape (n_transformed_dimensions,)
        Per-element upper bounds, broadcastable against ``x``.

    Returns
    -------
    ndarray of shape (..., n_transformed_dimensions)
        Values mapped into ``(lower_bounds, upper_bounds)`` element-wise.
    """

    return lower_bounds + (upper_bounds - lower_bounds) * expit(x)


def bounded_logit(
    x: NDArray[np.float64],
    lower_bounds: NDArray[np.float64],
    upper_bounds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Map values from per-element open intervals ``(lower_bounds, upper_bounds)`` to the real line.

    Applies the generalised logit function element-wise:
    ``log((x - lower_bounds) / (upper_bounds - x))``.
    This is the inverse of ``bounded_sigmoid``. Small epsilon terms guard
    against division by zero and ``log(0)`` at the boundaries.

    Parameters
    ----------
    x : ndarray of shape (..., n_transformed_dimensions)
        Values in ``(lower_bounds, upper_bounds)`` element-wise.
    lower_bounds : ndarray of shape (n_transformed_dimensions,)
        Per-element lower bounds, broadcastable against ``x``.
    upper_bounds : ndarray of shape (n_transformed_dimensions,)
        Per-element upper bounds, broadcastable against ``x``.

    Returns
    -------
    ndarray of shape (..., n_transformed_dimensions)
        Values mapped into ``(-inf, +inf)``.
    """

    return np.log(((x - lower_bounds) / (upper_bounds - x + 1e-50)) + 1e-50)