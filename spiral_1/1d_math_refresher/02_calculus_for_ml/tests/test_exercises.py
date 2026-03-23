"""
Tests for Module 02: Calculus for ML

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    numerical_derivative,
    numerical_partial_derivatives,
    gradient_of_mse,
    chain_rule_demo,
    gradient_descent_1d,
    gradient_descent_2d,
)


# ---------------------------------------------------------------------------
# numerical_derivative
# ---------------------------------------------------------------------------

class TestNumericalDerivative:
    def test_x_squared(self):
        """d/dx(x^2) = 2x, at x=3 should be 6."""
        result = numerical_derivative(lambda x: x ** 2, 3.0)
        assert result == pytest.approx(6.0, abs=1e-5)

    def test_sin(self):
        """d/dx(sin(x)) = cos(x), at x=0 should be 1."""
        result = numerical_derivative(np.sin, 0.0)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_exp(self):
        """d/dx(exp(x)) = exp(x), at x=1 should be e."""
        result = numerical_derivative(np.exp, 1.0)
        assert result == pytest.approx(np.e, abs=1e-5)

    def test_constant(self):
        result = numerical_derivative(lambda x: 5.0, 42.0)
        assert result == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# numerical_partial_derivatives
# ---------------------------------------------------------------------------

class TestNumericalPartialDerivatives:
    def test_simple_sum(self):
        """f(x,y) = x + y, grad = [1, 1]."""
        def f(v):
            return v[0] + v[1]
        grad = numerical_partial_derivatives(f, np.array([1.0, 2.0]))
        np.testing.assert_allclose(grad, [1.0, 1.0], atol=1e-5)

    def test_quadratic(self):
        """f(x,y) = x^2 + 3y^2, grad = [2x, 6y]."""
        def f(v):
            return v[0] ** 2 + 3 * v[1] ** 2
        grad = numerical_partial_derivatives(f, np.array([2.0, 1.0]))
        np.testing.assert_allclose(grad, [4.0, 6.0], atol=1e-5)

    def test_3d(self):
        """f(x,y,z) = x*y*z, grad at (2,3,4) = [12, 8, 6]."""
        def f(v):
            return v[0] * v[1] * v[2]
        grad = numerical_partial_derivatives(f, np.array([2.0, 3.0, 4.0]))
        np.testing.assert_allclose(grad, [12.0, 8.0, 6.0], atol=1e-5)


# ---------------------------------------------------------------------------
# gradient_of_mse
# ---------------------------------------------------------------------------

class TestGradientOfMse:
    def test_matches_numerical(self):
        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        w = np.random.randn(3)

        analytical = gradient_of_mse(X, y, w)

        def mse(w_):
            return np.mean((X @ w_ - y) ** 2)

        numerical = numerical_partial_derivatives(mse, w)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5)

    def test_at_optimum(self):
        """At the optimal w, gradient should be near zero."""
        np.random.seed(0)
        X = np.random.randn(100, 2)
        w_true = np.array([3.0, -1.0])
        y = X @ w_true
        grad = gradient_of_mse(X, y, w_true)
        np.testing.assert_allclose(grad, np.zeros(2), atol=1e-10)

    def test_shape(self):
        X = np.random.randn(10, 4)
        y = np.random.randn(10)
        w = np.random.randn(4)
        grad = gradient_of_mse(X, y, w)
        assert grad.shape == (4,)


# ---------------------------------------------------------------------------
# chain_rule_demo
# ---------------------------------------------------------------------------

class TestChainRuleDemo:
    def test_at_zero(self):
        result = chain_rule_demo(0.0)
        assert result["f"] == pytest.approx(0.0, abs=1e-10)
        assert result["df_dx_analytical"] == pytest.approx(0.0, abs=1e-10)

    def test_at_sqrt_pi_over_2(self):
        x = np.sqrt(np.pi / 2)
        result = chain_rule_demo(x)
        assert result["f"] == pytest.approx(1.0, abs=1e-10)  # sin(pi/2) = 1

    def test_analytical_matches_numerical(self):
        result = chain_rule_demo(1.5)
        assert result["df_dx_analytical"] == pytest.approx(
            result["df_dx_numerical"], abs=1e-5
        )

    def test_returns_dict_keys(self):
        result = chain_rule_demo(1.0)
        assert "f" in result
        assert "df_dx_analytical" in result
        assert "df_dx_numerical" in result


# ---------------------------------------------------------------------------
# gradient_descent_1d
# ---------------------------------------------------------------------------

class TestGradientDescent1d:
    def test_quadratic_minimum(self):
        """f(x) = (x-3)^2, minimum at x=3."""
        f = lambda x: (x - 3) ** 2
        df = lambda x: 2 * (x - 3)
        x_final, history = gradient_descent_1d(f, df, x0=0.0, lr=0.1)
        assert x_final == pytest.approx(3.0, abs=1e-4)

    def test_history_recorded(self):
        f = lambda x: x ** 2
        df = lambda x: 2 * x
        _, history = gradient_descent_1d(f, df, x0=5.0, lr=0.1, max_iter=10)
        assert len(history) > 0
        assert history[0] == 5.0

    def test_converges_early(self):
        """With tight tol, should stop before max_iter."""
        f = lambda x: x ** 2
        df = lambda x: 2 * x
        _, history = gradient_descent_1d(
            f, df, x0=0.001, lr=0.1, max_iter=10000, tol=1e-6
        )
        assert len(history) < 10000


# ---------------------------------------------------------------------------
# gradient_descent_2d
# ---------------------------------------------------------------------------

class TestGradientDescent2d:
    def test_quadratic_bowl(self):
        """f(x,y) = x^2 + y^2, minimum at (0,0)."""
        f = lambda v: v[0] ** 2 + v[1] ** 2
        grad_f = lambda v: np.array([2 * v[0], 2 * v[1]])
        x_final, _ = gradient_descent_2d(
            f, grad_f, x0=np.array([5.0, -3.0]), lr=0.1
        )
        np.testing.assert_allclose(x_final, [0.0, 0.0], atol=1e-4)

    def test_shifted_minimum(self):
        """f(x,y) = (x-1)^2 + (y+2)^2, minimum at (1,-2)."""
        f = lambda v: (v[0] - 1) ** 2 + (v[1] + 2) ** 2
        grad_f = lambda v: np.array([2 * (v[0] - 1), 2 * (v[1] + 2)])
        x_final, _ = gradient_descent_2d(
            f, grad_f, x0=np.array([0.0, 0.0]), lr=0.1
        )
        np.testing.assert_allclose(x_final, [1.0, -2.0], atol=1e-4)

    def test_history_shape(self):
        f = lambda v: v[0] ** 2 + v[1] ** 2
        grad_f = lambda v: np.array([2 * v[0], 2 * v[1]])
        _, history = gradient_descent_2d(
            f, grad_f, x0=np.array([1.0, 1.0]), lr=0.1, max_iter=5
        )
        assert len(history) > 0
        assert history[0].shape == (2,)
