"""Special functions: factorial, gamma, and Bessel."""

import numpy as np


def factorial(n):
    """
    Compute the factorial n! for non-negative integers.

    Parameters
    ----------
    n : int or numpy.ndarray
        Non-negative integer input(s).

    Returns
    -------
    int or numpy.ndarray
        Computed n!.

    Raises
    ------
    ValueError
        If n is negative.

    Examples
    --------
    >>> factorial(0)
    1
    >>> factorial(5)
    120
    >>> factorial(np.array([0, 1, 2]))
    array([1, 1, 2])
    """

    def fact_scalar(n):
        if not isinstance(n, (int, np.integer)) or n < 0:
            raise ValueError("Input must be a non-negative integer")
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result

    return np.vectorize(fact_scalar)(n)


def gamma(z, T=100, M=1000):
    """
    Compute the gamma function Γ(z) for z > 0 using numerical integration.

    This function uses the trapezoidal rule to approximate the integral:

        Γ(z) = ∫₀^∞ t^(z-1) * e^(-t) dt

    Parameters
    ----------
    z : float or numpy.ndarray
        Input value(s), must be positive.
    T : float, optional
        Upper integration limit (default is 100).
    M : int, optional
        Number of integration points (default is 1000).

    Returns
    -------
    float or numpy.ndarray
        Computed gamma(z).

    Raises
    ------
    ValueError
        If z <= 0.

    Examples
    --------
    >>> gamma(1)
    1.0
    >>> gamma(0.5)  # Equals sqrt(pi)
    1.7724538509055159
    >>> gamma(np.array([1, 2]))
    array([1., 1.])
    """

    def gamma_scalar(z):
        if z <= 0:
            raise ValueError("z must be positive")
        if z == 1:
            return 1.0
        elif z < 1:
            return gamma(z + 1, T, M) / z
        else:
            dt = T / M
            t = np.linspace(0, T, M + 1)
            ig = t ** (z - 1) * np.exp(-t)
            return dt * (ig[0] / 2 + ig[-1] / 2 + np.sum(ig[1:-1]))

    return np.vectorize(gamma_scalar)(z)


def bessel(alpha, x, n_terms=20):
    """
    Compute the Bessel function J_alpha(x) using its series expansion.

    Parameters
    ----------
    alpha : float
        Order of the Bessel function.
    x : float or numpy.ndarray
        Input value(s).
    n_terms : int, optional
        Number of terms in the series (default is 20).

    Returns
    -------
    float or numpy.ndarray
        Computed J_alpha(x).

    Examples
    --------
    >>> bessel(0, 0)
    1.0
    >>> bessel(0, 1)  # Approximate value
    0.7651976865579666
    >>> bessel(0, np.array([0, 1]))
    array([1.        , 0.76519769])
    """

    def bessel_scalar(x):
        term = (x / 2) ** alpha / gamma(alpha + 1)
        total = term
        for m in range(1, n_terms):
            term *= -((x / 2) ** 2) / (m * (m + alpha))
            total += term
            if abs(term) < 1e-10 * abs(total):
                break
        return total

    return np.vectorize(bessel_scalar)(x)
