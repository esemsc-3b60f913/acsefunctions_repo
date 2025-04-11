"""Transcendental functions computed via Taylor series."""

import numpy as np
from .utils import series_sum


def exp(x, n_terms=50):
    """
    Compute the exponential function e^x using Taylor series.

    The Taylor series is: e^x = sum_{n=0}^infty x^n / n!.
    Truncated at n_terms.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s).
    n_terms : int, optional
        Number of terms in the series (default is 50).

    Returns
    -------
    float or numpy.ndarray
        Computed e^x.

    Examples
    --------
    >>> exp(0)
    1.0
    >>> exp(1)
    2.718281828459045
    >>> exp(np.array([-1, 0, 1]))
    array([0.36787944, 1.        , 2.71828183])
    """

    def exp_scalar(x):
        return series_sum(lambda n: x**n, lambda n: 1 if n == 0 else n, n_terms)

    return np.vectorize(exp_scalar)(x)


def sinh(x, n_terms=50):
    """
    Compute the hyperbolic sine sinh(x) using Taylor series.

    The Taylor series is: sinh(x) = sum_{n=0}^infty x^(2n+1) / (2n+1)!.
    Truncated at n_terms.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s).
    n_terms : int, optional
        Number of terms in the series (default is 50).

    Returns
    -------
    float or numpy.ndarray
        Computed sinh(x).

    Examples
    --------
    >>> sinh(0)
    0.0
    >>> sinh(1)
    1.1752011936438014
    >>> sinh(np.array([0, 1]))
    array([0.        , 1.17520119])
    """

    def sinh_scalar(x):
        return series_sum(lambda n: x ** (2 * n + 1), lambda n: 2 * n + 1, n_terms)

    return np.vectorize(sinh_scalar)(x)


def cosh(x, n_terms=20):
    """
    Compute the hyperbolic cosine cosh(x) using Taylor series.

    The Taylor series is: cosh(x) = sum_{n=0}^infty x^(2n) / (2n)!.
    Truncated at n_terms.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s).
    n_terms : int, optional
        Number of terms in the series (default is 50).

    Returns
    -------
    float or numpy.ndarray
        Computed cosh(x).

    Examples
    --------
    >>> cosh(0)
    1.0
    >>> cosh(1)
    1.5430806348152437
    >>> cosh(np.array([0, 1]))
    array([1.        , 1.54308063])
    """

    def cosh_scalar(x):
        return series_sum(lambda n: x ** (2 * n), lambda n: 2 * n, n_terms)

    return np.vectorize(cosh_scalar)(x)


def tanh(x, n_terms=50):
    """
    Compute the hyperbolic tangent tanh(x) as sinh(x) / cosh(x).

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s).
    n_terms : int, optional
        Number of terms in the series for sinh and cosh (default is 50).

    Returns
    -------
    float or numpy.ndarray
        Computed tanh(x).

    Examples
    --------
    >>> tanh(0)
    0.0
    >>> tanh(1)
    0.7615941559557649
    >>> tanh(np.array([0, 1]))
    array([0.        , 0.76159416])
    """
    s = sinh(x, n_terms)
    c = cosh(x, n_terms)
    return s / c
