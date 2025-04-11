"""Test docstrings with doctest."""

import doctest
from acsefunctions import transcendental, special


def test_docstrings():
    doctest.testmod(transcendental)
    doctest.testmod(special)
