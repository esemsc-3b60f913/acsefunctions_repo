"""Utility functions for series computations."""


def series_sum(power_func, fact_func, n_terms):
    """Compute a series sum iteratively."""
    total = 0
    term = 1  # For n=0
    total += term
    for n in range(1, n_terms):
        term *= power_func(n) / fact_func(n)
        total += term
    return total
