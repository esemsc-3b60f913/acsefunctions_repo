import numpy as np


def exp(x, n_terms=20):
    """
    Compute the exponential function e^x using a Taylor series
    approximation.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s) for which to compute e^x.
    n_terms : int, optional
        Number of terms to use in the Taylor series (default is 20).

    Returns
    -------
    float or numpy.ndarray
        Computed e^x for the input(s).

    Examples
    --------
    >>> exp(0)
    1.0
    >>> exp(1)
    2.718281828459045
    >>> exp(np.array([0, 1, 2]))
    array([1.        , 2.71828183, 7.3890561 ])
    """

    def exp_scalar(val):
        total = 1.0  # n=0: x^0 / 0! = 1
        term = 1.0
        for n in range(1, n_terms):
            term *= val / n
            total += term
            if abs(term) < 1e-15:
                break
        return total

    return np.vectorize(exp_scalar)(x)


def sinh(x, n_terms=20):
    """
    Compute the hyperbolic sine function sinh(x) using a Taylor series
    approximation.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s) for which to compute sinh(x).
    n_terms : int, optional
        Number of terms to use in the Taylor series (default is 20).

    Returns
    -------
    float or numpy.ndarray
        Computed sinh(x) for the input(s).

    Examples
    --------
    >>> sinh(0)
    0.0
    >>> sinh(1)
    1.1752011936438014
    >>> sinh(np.array([0, 1, 2]))
    array([0.        , 1.17520119, 3.62686041])
    """

    def sinh_scalar(val):
        total = val  # n=0: x^1 / 1! = x
        term = val
        for n in range(1, n_terms):
            term *= (val**2) / ((2 * n + 1) * (2 * n))
            total += term
            if abs(term) < 1e-15:
                break
        return total

    return np.vectorize(sinh_scalar)(x)


def cosh(x, n_terms=20):
    """
    Compute the hyperbolic cosine function cosh(x) using a Taylor series
    approximation.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s) for which to compute cosh(x).
    n_terms : int, optional
        Number of terms to use in the Taylor series (default is 20).

    Returns
    -------
    float or numpy.ndarray
        Computed cosh(x) for the input(s).

    Examples
    --------
    >>> cosh(0)
    1.0
    >>> cosh(1)
    1.5430806348152437
    >>> cosh(np.array([0, 1, 2]))
    array([1.        , 1.54308063, 3.76219569])
    """

    def cosh_scalar(val):
        total = 1.0  # n=0: x^0 / 0! = 1
        term = 1.0
        for n in range(1, n_terms):
            term *= (val**2) / ((2 * n) * (2 * n - 1))
            total += term
            if abs(term) < 1e-15:
                break
        return total

    return np.vectorize(cosh_scalar)(x)


def tanh(x, n_terms=20):
    """
    Compute the hyperbolic tangent function tanh(x) as sinh(x) / cosh(x).

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s) for which to compute tanh(x).
    n_terms : int, optional
        Number of terms to use in the Taylor series for sinh and cosh (default is 20).

    Returns
    -------
    float or numpy.ndarray
        Computed tanh(x) for the input(s).

    Raises
    ------
    ZeroDivisionError
        If cosh(x) equals zero, which can occur in rare numerical edge cases.

    Examples
    --------
    >>> tanh(0)
    0.0
    >>> tanh(1)
    0.7615941559557649
    >>> tanh(np.array([0, 1, 2]))
    array([0.        , 0.76159416, 0.96402758])
    """

    s = sinh(x, n_terms)
    c = cosh(x, n_terms)
    return s / c
