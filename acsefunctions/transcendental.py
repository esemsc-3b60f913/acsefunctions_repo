"""Transcendental functions computed via Taylor series."""

import numpy as np


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

    def exp_scalar(val):
        total = 1.0  # Term for n=0
        term = 1.0
        for n in range(1, n_terms):
            term *= val / n  # term_n = term_{n-1} * x / n
            total += term
            if abs(term) < 1e-15:  # Early stopping for convergence
                break
        return total

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

    def sinh_scalar(val):
        total = val  # Term for n=0: x^1 / 1!
        term = val
        for n in range(1, n_terms):
            term *= (val**2) / (
                (2 * n + 1) * (2 * n)
            )  # term_n = term_{n-1} * x^2 / ((2n+1)(2n))
            total += term
            if abs(term) < 1e-15:  # Early stopping for convergence
                break
        return total

    return np.vectorize(sinh_scalar)(x)


def cosh(x, n_terms=50):
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

    def cosh_scalar(val):
        total = 1.0  # Term for n=0: x^0 / 0!
        term = 1.0
        for n in range(1, n_terms):
            term *= (val**2) / (
                (2 * n) * (2 * n - 1)
            )  # term_n = term_{n-1} * x^2 / (2n(2n-1))
            total += term
            if abs(term) < 1e-15:  # Early stopping for convergence
                break
        return total

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
    return sinh(x, n_terms) / cosh(x, n_terms)
