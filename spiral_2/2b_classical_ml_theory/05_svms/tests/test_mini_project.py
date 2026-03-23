"""
Tests for the Kernel SVM mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    generate_linearly_separable,
    generate_concentric_circles,
    generate_two_moons,
)


class TestDataGeneration:
    def test_linearly_separable_shapes(self):
        X, y = generate_linearly_separable(200)
        assert X.shape == (200, 2)
        assert y.shape == (200,)
        assert set(np.unique(y)) == {-1, 1}

    def test_circles_shapes(self):
        X, y = generate_concentric_circles(200)
        assert X.shape == (200, 2)

    def test_moons_shapes(self):
        X, y = generate_two_moons(200)
        assert X.shape == (200, 2)
