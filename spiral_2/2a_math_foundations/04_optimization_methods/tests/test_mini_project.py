"""
Tests for the Optimizer Shootout mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    convex_quadratic,
    convex_quadratic_grad,
    rastrigin,
    rastrigin_grad,
)


class TestConvexQuadratic:
    def test_minimum_at_origin(self):
        assert convex_quadratic(np.array([0.0, 0.0])) == 0.0

    def test_positive_elsewhere(self):
        assert convex_quadratic(np.array([1.0, 1.0])) > 0

    def test_gradient_at_origin_is_zero(self):
        grad = convex_quadratic_grad(np.array([0.0, 0.0]))
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-10)


class TestRastrigin:
    def test_minimum_at_origin(self):
        assert rastrigin(np.array([0.0, 0.0])) == 0.0

    def test_positive_elsewhere(self):
        assert rastrigin(np.array([0.5, 0.5])) > 0

    def test_gradient_at_origin(self):
        grad = rastrigin_grad(np.array([0.0, 0.0]))
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-10)
