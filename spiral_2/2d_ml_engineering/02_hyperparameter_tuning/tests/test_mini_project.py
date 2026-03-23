"""
Tests for the Bayesian Optimization mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    branin_function,
)


class TestBranin:
    def test_known_minimum(self):
        """Branin has known global minimum ~0.397887."""
        # One of the three global minima: (pi, 2.275)
        val = branin_function(np.array([np.pi, 2.275]))
        np.testing.assert_allclose(val, 0.397887, atol=0.01)
