"""Tests for special functions."""

import numpy as np
from acsefunctions import factorial, gamma, bessel
import pytest


def test_factorial():
    assert factorial(0) == 1
    assert factorial(5) == 120
    assert np.all(factorial(np.array([0, 1, 2])) == [1, 1, 2])
    with pytest.raises(ValueError):
        factorial(-1)


def test_gamma():
    assert gamma(1) == 1.0
    assert np.allclose(gamma(0.5), np.sqrt(np.pi), rtol=1e-2)
    assert np.allclose(gamma(np.array([1, 2])), [1, 1], rtol=1e-2)
    with pytest.raises(ValueError):
        gamma(0)


def test_bessel():
    assert bessel(0, 0) == 1.0
    assert np.allclose(bessel(0, 1), 0.7651976865579666, rtol=1e-2)
    assert np.allclose(bessel(0, np.array([0, 1])), [1, 0.7651976865579666], rtol=1e-2)
