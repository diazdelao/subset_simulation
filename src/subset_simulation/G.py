from functools import lru_cache

import numpy as np


def _require_dimension(X, expected_dim, name):
    """Validate that a test function receives the dimension it expects."""
    if X.shape[-1] != expected_dim:
        raise ValueError(f"{name} requires exactly {expected_dim} dimensions.")


def _ackley(X):
    """Ackley function with global minimum at x = (0, ..., 0)."""
    n = X.shape[-1]
    term1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(X**2, axis=-1) / n))
    term2 = -np.exp(np.sum(np.cos(2 * np.pi * X), axis=-1) / n)
    return -(term1 + term2 + 20 + np.e)


def _sphere(X):
    """Sphere function with global minimum at x = (0, ..., 0)."""
    return -np.sum(X**2, axis=-1)


def _rastrigin(X):
    """Rastrigin function with global minimum at x = (0, ..., 0)."""
    n = X.shape[-1]
    return -(10 * n + np.sum(X**2 - 10 * np.cos(2 * np.pi * X), axis=-1))


def _rosenbrock(X):
    """Rosenbrock function with global minimum at x = (1, ..., 1)."""
    if X.shape[-1] < 2:
        raise ValueError("Rosenbrock requires at least 2 dimensions.")
    return -np.sum(
        100 * (X[..., 1:] - X[..., :-1] ** 2) ** 2 + (1 - X[..., :-1]) ** 2,
        axis=-1,
    )


def _bukin6(X):
    """
    Bukin N.6 function.

    The SFU page notes many local minima along a ridge; the standard global
    minimum is at (-10, 1).
    Source: https://www.sfu.ca/~ssurjano/bukin6.html
    """
    _require_dimension(X, 2, "Bukin N.6")
    x1 = X[..., 0]
    x2 = X[..., 1]
    term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2))
    term2 = 0.01 * np.abs(x1 + 10)
    return -(term1 + term2)


def _crossintray(X):
    """
    Cross-in-Tray function.

    The SFU page reports multiple global minima; the standard four are at
    (±1.34941, ±1.34941).
    Source: https://www.sfu.ca/~ssurjano/crossit.html
    """
    _require_dimension(X, 2, "Cross-in-Tray")
    x1 = X[..., 0]
    x2 = X[..., 1]
    fact1 = np.sin(x1) * np.sin(x2)
    fact2 = np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
    return 0.0001 * (np.abs(fact1 * fact2) + 1.0) ** 0.1


def _eggholder(X):
    """
    Eggholder function.

    The SFU page notes many local minima; the standard global minimum is near
    (512, 404.2319).
    Source: https://www.sfu.ca/~ssurjano/egg.html
    """
    _require_dimension(X, 2, "Eggholder")
    x1 = X[..., 0]
    x2 = X[..., 1]
    term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    return -(term1 + term2)


def _holder(X):
    """
    Holder Table function.

    The SFU page reports four global minima, at approximately
    (±8.05502, ±9.66459).
    Source: https://www.sfu.ca/~ssurjano/holder.html
    """
    _require_dimension(X, 2, "Holder Table")
    x1 = X[..., 0]
    x2 = X[..., 1]
    fact1 = np.sin(x1) * np.cos(x2)
    fact2 = np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))
    return np.abs(fact1 * fact2)


def _langermann(X):
    """
    Langermann function with the standard SFU 2D parameters.

    The SFU page describes many unevenly distributed local minima; this
    implementation uses the recommended d = 2, m = 5 parameter set.
    Source: https://www.sfu.ca/~ssurjano/langer.html
    """
    _require_dimension(X, 2, "Langermann")
    c = np.array([1.0, 2.0, 5.0, 2.0, 3.0])
    A = np.array([[3.0, 5.0], [5.0, 2.0], [2.0, 1.0], [1.0, 4.0], [7.0, 9.0]])
    diffs = X[..., np.newaxis, :] - A
    inner = np.sum(diffs**2, axis=-1)
    values = c * np.exp(-inner / np.pi) * np.cos(np.pi * inner)
    return -np.sum(values, axis=-1)


def _levy(X):
    """
    Levy function with global minimum at x = (1, ..., 1).

    The SFU page presents the d-dimensional Levy benchmark on [-10, 10]^d.
    Source: https://www.sfu.ca/~ssurjano/levy.html
    """
    w = 1 + (X - 1) / 4
    term1 = np.sin(np.pi * w[..., 0]) ** 2
    term3 = (w[..., -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[..., -1]) ** 2)
    wi = w[..., :-1]
    middle = np.sum((wi - 1) ** 2 * (1 + 10 * np.sin(np.pi * wi + 1) ** 2), axis=-1)
    return -(term1 + middle + term3)


_FUNCTIONS = {
    "ackley": _ackley,
    "sphere": _sphere,
    "rastrigin": _rastrigin,
    "rosenbrock": _rosenbrock,
    "bukin6": _bukin6,
    "crossintray": _crossintray,
    "eggholder": _eggholder,
    "holder": _holder,
    "langermann": _langermann,
    "levy": _levy,
}

_CURRENT_FUNCTION_NAME = "ackley"


def get_available_test_functions():
    """Return the names of the registered test functions."""
    return tuple(_FUNCTIONS)


def get_test_function_name():
    """Return the name of the active test function."""
    return _CURRENT_FUNCTION_NAME


def set_test_function(name):
    """Select which registered test function G() and G_batch() should use."""
    normalized_name = str(name).strip().lower()
    if normalized_name not in _FUNCTIONS:
        available = ", ".join(get_available_test_functions())
        raise ValueError(
            f"Unknown test function '{name}'. Available functions: {available}."
        )

    global _CURRENT_FUNCTION_NAME
    _CURRENT_FUNCTION_NAME = normalized_name
    _G_cached.cache_clear()


def _evaluate_active_function(X):
    X = np.asarray(X, dtype=float)
    if X.shape[-1] < 1:
        raise ValueError("Input must have at least one dimension.")
    return _FUNCTIONS[_CURRENT_FUNCTION_NAME](X)


def _G_internal(X_tuple):
    # Input has to be a tuple to be hashable, so we first convert it to an array.
    X = np.asarray(X_tuple, dtype=float)
    return float(_evaluate_active_function(X))


def G(X_tuple):
    d = len(X_tuple)
    if d > 2:
        return _G_cached(_CURRENT_FUNCTION_NAME, tuple(X_tuple))
    return _G_internal(X_tuple)


def G_batch(X):
    """
    Evaluate the active test function for samples stored row-wise with shape
    (n_samples, d).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("G_batch expects a 2D array with shape (n_samples, d).")
    return _evaluate_active_function(X)


@lru_cache(maxsize=10000)
def _G_cached(function_name, X_tuple):
    return float(_FUNCTIONS[function_name](np.asarray(X_tuple, dtype=float)))
