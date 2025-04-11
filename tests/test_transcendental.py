"""Tests for transcendental functions."""

import numpy as np
from acsefunctions import exp, sinh, cosh, tanh


def test_exp():
    assert exp(0) == 1.0
    assert np.allclose(exp(1), 2.718281828459045, rtol=1e-8)
    assert np.allclose(exp(np.array([-1, 0, 1])), np.exp([-1, 0, 1]), rtol=1e-8)


def test_sinh():
    assert sinh(0) == 0.0
    assert np.allclose(sinh(1), 1.1752011936438014, rtol=1e-8)
    assert np.allclose(sinh(np.array([0, 1])), np.sinh([0, 1]), rtol=1e-8)


def test_cosh():
    assert cosh(0) == 1.0
    assert np.allclose(cosh(1), 1.5430806348152437, rtol=1e-8)
    assert np.allclose(cosh(np.array([0, 1])), np.cosh([0, 1]), rtol=1e-8)


def test_tanh():
    assert tanh(0) == 0.0
    assert np.allclose(tanh(1), 0.7615941559557649, rtol=1e-8)
    assert np.allclose(tanh(np.array([0, 1])), np.tanh([0, 1]), rtol=1e-8)
