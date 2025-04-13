import numpy as np


def exp(x, n_terms=20):
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
    s = sinh(x, n_terms)
    c = cosh(x, n_terms)
    return s / c
