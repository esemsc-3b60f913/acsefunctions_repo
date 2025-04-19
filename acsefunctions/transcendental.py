import numpy as np


def exp(x, n_terms=20):
    """Compute the exponential function e^x using a Taylor series.

    Parameters
    ----------
    x : array_like
        Input values.
    n_terms : int, optional
        Number of terms to use in the Taylor series expansion (default is 20).

    Returns
    -------
    ndarray
        Exponential of the input values, computed element-wise.

    Notes
    -----
    The Taylor series for e^x is:
        e^x = 1 + x/1! + x^2/2! + x^3/3! + ...
    The computation stops early if a term's magnitude is less than 1e-15.
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
    """Compute the hyperbolic sine function sinh(x) using a Taylor series.

    Parameters
    ----------
    x : array_like
        Input values.
    n_terms : int, optional
        Number of terms to use in the Taylor series expansion (default is 20).

    Returns
    -------
    ndarray
        Hyperbolic sine of the input values, computed element-wise.

    Notes
    -----
    The Taylor series for sinh(x) is:
        sinh(x) = x + x^3/3! + x^5/5! + x^7/7! + ...
    The computation stops early if a term's magnitude is less than 1e-15.
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
    """Compute the hyperbolic cosine function cosh(x) using a Taylor series.

    Parameters
    ----------
    x : array_like
        Input values.
    n_terms : int, optional
        Number of terms to use in the Taylor series expansion (default is 20).

    Returns
    -------
    ndarray
        Hyperbolic cosine of the input values, computed element-wise.

    Notes
    -----
    The Taylor series for cosh(x) is:
        cosh(x) = 1 + x^2/2! + x^4/4! + x^6/6! + ...
    The computation stops early if a term's magnitude is less than 1e-15.
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
    """Compute the hyperbolic tangent function tanh(x) using sinh(x) and cosh(x).

    Parameters
    ----------
    x : array_like
        Input values.
    n_terms : int, optional
        Number of terms to use in the Taylor series expansion for sinh and cosh (default is 20).

    Returns
    -------
    ndarray
        Hyperbolic tangent of the input values, computed element-wise as sinh(x) / cosh(x).
    """
    s = sinh(x, n_terms)
    c = cosh(x, n_terms)
    return s / c
