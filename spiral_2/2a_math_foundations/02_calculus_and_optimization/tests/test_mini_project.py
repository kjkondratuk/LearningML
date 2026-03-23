"""
Tests for the Gradient Descent Visualizer mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    quadratic_bowl,
    quadratic_bowl_grad,
    rosenbrock,
    rosenbrock_grad,
    saddle_point,
    saddle_point_grad,
    create_contour_data,
)


class TestQuadraticBowl:
    def test_minimum_at_origin(self):
        assert quadratic_bowl(np.array([0.0, 0.0])) == 0.0

    def test_gradient_at_origin(self):
        grad = quadratic_bowl_grad(np.array([0.0, 0.0]))
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-10)

    def test_ill_conditioned(self):
        x = np.array([1.0, 1.0])
        val = quadratic_bowl(x, condition_number=100.0)
        expected = 0.5 * (1.0 + 100.0)
        np.testing.assert_allclose(val, expected, atol=1e-10)


class TestRosenbrock:
    def test_minimum_at_one_one(self):
        assert rosenbrock(np.array([1.0, 1.0])) == 0.0

    def test_gradient_at_minimum(self):
        grad = rosenbrock_grad(np.array([1.0, 1.0]))
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-10)


class TestSaddlePoint:
    def test_value_at_origin(self):
        assert saddle_point(np.array([0.0, 0.0])) == 0.0

    def test_gradient_at_origin(self):
        grad = saddle_point_grad(np.array([0.0, 0.0]))
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-10)

    def test_positive_along_x(self):
        assert saddle_point(np.array([1.0, 0.0])) > 0

    def test_negative_along_y(self):
        assert saddle_point(np.array([0.0, 1.0])) < 0


class TestContourData:
    def test_shapes(self):
        f = lambda x: float(x[0] ** 2 + x[1] ** 2)
        X, Y, Z = create_contour_data(f, (-1, 1), (-1, 1), resolution=50)
        assert X.shape == (50, 50)
        assert Y.shape == (50, 50)
        assert Z.shape == (50, 50)
