"""Utility functions for series computations."""


def series_sum(power_func, fact_func, n_terms):
    """Compute a series sum iteratively, preventing overflow."""
    total = 0.0
    term = 1.0  # For n=0, using float for consistency
    total += term
    for n in range(1, n_terms):
        next_term = term * power_func(n) / fact_func(n)
        if (
            abs(next_term) > 1e308 or abs(next_term) < 1e-15
        ):  # Break on overflow risk or convergence
            break
        term = next_term
        total += term
    return total
