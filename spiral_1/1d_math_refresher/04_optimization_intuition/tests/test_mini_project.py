"""
Tests for Mini-Project: Visualize Optimization Landscape

Run: pytest tests/test_mini_project.py -v
"""

import numpy as np
import pytest

from mini_project import (
    rosenbrock,
    rosenbrock_grad,
    beale,
    beale_grad,
    visualize_optimization_landscape,
)


# ---------------------------------------------------------------------------
# rosenbrock
# ---------------------------------------------------------------------------

class TestRosenbrock:
    def test_minimum_at_1_1(self):
        assert rosenbrock(np.array([1.0, 1.0])) == pytest.approx(0.0)

    def test_known_value(self):
        # f(0,0) = (1-0)^2 + 100*(0 - 0)^2 = 1
        assert rosenbrock(np.array([0.0, 0.0])) == pytest.approx(1.0)

    def test_positive_away_from_minimum(self):
        assert rosenbrock(np.array([2.0, 3.0])) > 0


# ---------------------------------------------------------------------------
# rosenbrock_grad
# ---------------------------------------------------------------------------

class TestRosenbrockGrad:
    def test_zero_at_minimum(self):
        grad = rosenbrock_grad(np.array([1.0, 1.0]))
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-10)

    def test_matches_numerical(self):
        x = np.array([0.5, 0.5])
        analytical = rosenbrock_grad(x)
        h = 1e-7
        numerical = np.zeros(2)
        for i in range(2):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            numerical[i] = (rosenbrock(x_plus) - rosenbrock(x_minus)) / (2 * h)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5)


# ---------------------------------------------------------------------------
# beale
# ---------------------------------------------------------------------------

class TestBeale:
    def test_minimum_at_3_05(self):
        assert beale(np.array([3.0, 0.5])) == pytest.approx(0.0, abs=1e-10)

    def test_positive_away_from_minimum(self):
        assert beale(np.array([0.0, 0.0])) > 0


# ---------------------------------------------------------------------------
# beale_grad
# ---------------------------------------------------------------------------

class TestBealeGrad:
    def test_near_zero_at_minimum(self):
        grad = beale_grad(np.array([3.0, 0.5]))
        np.testing.assert_allclose(grad, [0.0, 0.0], atol=1e-4)


# ---------------------------------------------------------------------------
# visualize_optimization_landscape
# ---------------------------------------------------------------------------

class TestVisualizeOptimizationLandscape:
    def test_returns_expected_structure(self):
        result = visualize_optimization_landscape(
            func_name="rosenbrock",
            methods=["vanilla", "momentum"],
            max_iter=100,
        )
        assert "vanilla" in result
        assert "momentum" in result
        for method, data in result.items():
            assert "trajectory" in data
            assert "final_point" in data
            assert "final_loss" in data
            assert "n_iterations" in data

    @pytest.mark.slow
    def test_rosenbrock_convergence(self):
        result = visualize_optimization_landscape(
            func_name="rosenbrock",
            methods=["adam"],
            lr=0.001,
            max_iter=10000,
        )
        assert result["adam"]["final_loss"] < 1.0

    def test_beale_function(self):
        result = visualize_optimization_landscape(
            func_name="beale",
            methods=["momentum"],
            max_iter=100,
        )
        assert "momentum" in result
        assert len(result["momentum"]["trajectory"]) > 0
