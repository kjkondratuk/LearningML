"""
Tests for Calculus and Optimization exercises.
"""

import numpy as np
import pytest

from exercises import (
    numerical_gradient,
    numerical_jacobian,
    numerical_hessian,
    taylor_approximate,
    gradient_descent,
    gradient_descent_with_momentum,
    check_convexity,
    lagrange_dual,
)

TOL = 1e-4


# ── numerical_gradient ────────────────────────────────────────────────────────

class TestNumericalGradient:
    def test_x_squared(self):
        """f(x) = x^2, grad = 2x."""
        f = lambda x: float(x[0] ** 2)
        grad = numerical_gradient(f, np.array([3.0]))
        np.testing.assert_allclose(grad, [6.0], atol=TOL)

    def test_sin(self):
        """f(x) = sin(x), grad = cos(x)."""
        f = lambda x: float(np.sin(x[0]))
        x = np.array([1.0])
        grad = numerical_gradient(f, x)
        np.testing.assert_allclose(grad, [np.cos(1.0)], atol=TOL)

    def test_multivariable(self):
        """f(x,y) = x^2 + y^2, grad = [2x, 2y]."""
        f = lambda x: float(x[0] ** 2 + x[1] ** 2)
        x = np.array([3.0, 4.0])
        grad = numerical_gradient(f, x)
        np.testing.assert_allclose(grad, [6.0, 8.0], atol=TOL)

    def test_quadratic_form(self):
        """f(x) = 0.5 x^T A x, grad = Ax."""
        A = np.array([[2, 1], [1, 3]], dtype=float)
        f = lambda x: float(0.5 * x @ A @ x)
        x = np.array([1.0, 2.0])
        grad = numerical_gradient(f, x)
        np.testing.assert_allclose(grad, A @ x, atol=TOL)


# ── numerical_jacobian ────────────────────────────────────────────────────────

class TestNumericalJacobian:
    def test_linear_function(self):
        """f(x) = Ax, Jacobian = A."""
        A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        f = lambda x: A @ x
        x = np.array([1.0, 1.0])
        J = numerical_jacobian(f, x)
        np.testing.assert_allclose(J, A, atol=TOL)

    def test_shape(self):
        f = lambda x: np.array([x[0] ** 2, x[1] ** 3, x[0] * x[1]])
        J = numerical_jacobian(f, np.array([2.0, 3.0]))
        assert J.shape == (3, 2)


# ── numerical_hessian ─────────────────────────────────────────────────────────

class TestNumericalHessian:
    def test_quadratic(self):
        """f(x) = 0.5 x^T A x => H = A (for symmetric A)."""
        A = np.array([[4, 1], [1, 3]], dtype=float)
        f = lambda x: float(0.5 * x @ A @ x)
        H = numerical_hessian(f, np.array([1.0, 2.0]))
        np.testing.assert_allclose(H, A, atol=1e-3)

    def test_symmetry(self):
        f = lambda x: float(np.sin(x[0]) * np.cos(x[1]) + x[0] ** 2 * x[1])
        H = numerical_hessian(f, np.array([1.0, 2.0]))
        np.testing.assert_allclose(H, H.T, atol=1e-3)


# ── taylor_approximate ────────────────────────────────────────────────────────

class TestTaylorApproximate:
    def test_first_order_exact_for_linear(self):
        """Taylor order-1 is exact for linear functions."""
        f = lambda x: float(3 * x[0] + 2 * x[1] + 1)
        x0 = np.array([0.0, 0.0])
        points = np.array([[1.0, 1.0], [2.0, 3.0]])
        approx = taylor_approximate(f, x0, order=1, points=points)
        exact = np.array([f(p) for p in points])
        np.testing.assert_allclose(approx, exact, atol=1e-3)

    def test_second_order_better_than_first(self):
        """For a cubic function, order-2 should be closer than order-1 near x0."""
        f = lambda x: float(x[0] ** 3 + x[1] ** 3)
        x0 = np.array([1.0, 1.0])
        points = np.array([[1.1, 1.1]])
        approx_1 = taylor_approximate(f, x0, order=1, points=points)
        approx_2 = taylor_approximate(f, x0, order=2, points=points)
        exact = np.array([f(p) for p in points])
        assert abs(approx_2[0] - exact[0]) <= abs(approx_1[0] - exact[0])


# ── gradient_descent ──────────────────────────────────────────────────────────

class TestGradientDescent:
    def test_converges_to_minimum_of_quadratic(self):
        """f(x) = x^2 + y^2, minimum at origin."""
        f = lambda x: float(x[0] ** 2 + x[1] ** 2)
        grad_f = lambda x: 2 * x
        traj = gradient_descent(f, grad_f, np.array([5.0, 5.0]), lr=0.1, n_steps=100)
        np.testing.assert_allclose(traj[-1], [0.0, 0.0], atol=1e-3)

    def test_trajectory_length(self):
        f = lambda x: float(x[0] ** 2)
        grad_f = lambda x: np.array([2 * x[0]])
        traj = gradient_descent(f, grad_f, np.array([1.0]), lr=0.1, n_steps=50)
        assert len(traj) == 51  # n_steps + 1 (including x0)

    def test_converges_on_rosenbrock(self):
        """GD should make progress toward (1,1) on Rosenbrock, even if slow."""
        def f(x):
            return float((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)
        def grad_f(x):
            dx = -2 * (1 - x[0]) + 200 * (x[1] - x[0] ** 2) * (-2 * x[0])
            dy = 200 * (x[1] - x[0] ** 2)
            return np.array([dx, dy])
        traj = gradient_descent(f, grad_f, np.array([-1.0, 1.0]), lr=0.001, n_steps=5000)
        # Should at least reduce loss
        assert f(traj[-1]) < f(traj[0])


# ── gradient_descent_with_momentum ────────────────────────────────────────────

class TestGDWithMomentum:
    def test_converges_faster_than_vanilla(self):
        """Momentum should converge in fewer effective steps on ill-conditioned quadratic."""
        A = np.diag([1.0, 100.0])
        f = lambda x: float(0.5 * x @ A @ x)
        grad_f = lambda x: A @ x
        x0 = np.array([10.0, 10.0])

        traj_gd = gradient_descent(f, grad_f, x0.copy(), lr=0.01, n_steps=200)
        traj_mom = gradient_descent_with_momentum(
            f, grad_f, x0.copy(), lr=0.01, momentum=0.9, n_steps=200
        )
        # Momentum should reach closer to 0
        assert f(traj_mom[-1]) < f(traj_gd[-1])

    def test_zero_momentum_matches_vanilla(self):
        f = lambda x: float(x[0] ** 2 + x[1] ** 2)
        grad_f = lambda x: 2 * x
        x0 = np.array([3.0, 4.0])
        traj_gd = gradient_descent(f, grad_f, x0.copy(), lr=0.1, n_steps=20)
        traj_mom = gradient_descent_with_momentum(
            f, grad_f, x0.copy(), lr=0.1, momentum=0.0, n_steps=20
        )
        np.testing.assert_allclose(traj_gd[-1], traj_mom[-1], atol=1e-10)


# ── check_convexity ───────────────────────────────────────────────────────────

class TestCheckConvexity:
    def test_quadratic_is_convex(self):
        f = lambda x: float(x[0] ** 2 + x[1] ** 2)
        assert check_convexity(f, (np.array([-5, -5]), np.array([5, 5]))) is True

    def test_non_convex(self):
        f = lambda x: float(np.sin(x[0]) + np.sin(x[1]))
        assert check_convexity(f, (np.array([-10, -10]), np.array([10, 10]))) is False


# ── lagrange_dual ─────────────────────────────────────────────────────────────

class TestLagrangeDual:
    def test_known_solution(self):
        """min x^2 + y^2 s.t. x + y = 1. Solution: x = y = 0.5."""
        f = lambda x: float(x[0] ** 2 + x[1] ** 2)
        g = lambda x: float(x[0] + x[1] - 1)
        x_star, lam = lagrange_dual(f, g, x_range=(-2.0, 2.0))
        np.testing.assert_allclose(x_star, [0.5, 0.5], atol=0.05)
