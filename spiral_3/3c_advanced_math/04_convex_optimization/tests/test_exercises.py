"""
Tests for Convex Optimization exercises.

Verifies: convexity detection, KKT conditions, proximal operator properties,
ISTA convergence, Fenchel conjugate.
"""

import pytest
import torch

from exercises import (
    admm_step,
    fenchel_conjugate_quadratic,
    kkt_conditions_check,
    proximal_gradient_descent,
    proximal_operator_l1,
    verify_convexity,
)


class TestConvexity:
    def test_quadratic_is_convex(self):
        def f(x):
            return (x ** 2).sum()
        points = torch.randn(20, 5)
        assert verify_convexity(f, points) is True

    def test_negative_quadratic_not_convex(self):
        def f(x):
            return -(x ** 2).sum()
        points = torch.randn(20, 5)
        assert verify_convexity(f, points) is False


class TestProximalL1:
    def test_produces_zeros(self):
        """Small values should be thresholded to exactly zero."""
        x = torch.tensor([0.5, -0.3, 0.1, -0.05])
        result = proximal_operator_l1(x, threshold=0.4)
        # Values with |x| < threshold should be 0
        assert result[2] == 0.0
        assert result[3] == 0.0

    def test_shrinks_large_values(self):
        """Large values should be shrunk by threshold."""
        x = torch.tensor([3.0, -3.0])
        result = proximal_operator_l1(x, threshold=1.0)
        expected = torch.tensor([2.0, -2.0])
        assert torch.allclose(result, expected)

    def test_zero_threshold_identity(self):
        x = torch.randn(10)
        result = proximal_operator_l1(x, threshold=0.0)
        assert torch.allclose(result, x)


class TestKKTConditions:
    def test_simple_qp_solution(self):
        """For min x^2 s.t. x >= 1, optimal is x*=1, lambda*=2."""
        x_star = torch.tensor([1.0])
        lambda_star = torch.tensor([2.0])
        f_grad = 2 * x_star  # grad of x^2
        g = [-(x_star - 1)]  # g(x) = -(x-1) <= 0 (reformulated as x >= 1)
        g_grad = [torch.tensor([-1.0])]
        result = kkt_conditions_check(x_star, lambda_star, f_grad, g, g_grad)
        assert all(result.values()), f"KKT should hold: {result}"


class TestProximalGradientDescent:
    def test_lasso_convergence(self):
        """ISTA should converge on a LASSO problem."""
        torch.manual_seed(42)
        n, d = 50, 10
        X = torch.randn(n, d)
        w_true = torch.zeros(d)
        w_true[:3] = torch.tensor([1.0, -2.0, 0.5])
        y = X @ w_true + 0.1 * torch.randn(n)
        lambda_reg = 0.1

        def f_grad(w):
            return X.T @ (X @ w - y) / n

        def prox_g(w, lr):
            return proximal_operator_l1(w, threshold=lr * lambda_reg)

        w_final, losses = proximal_gradient_descent(f_grad, prox_g, torch.zeros(d), lr=0.01, iterations=1000)
        # Should have some zero entries (sparsity)
        assert (w_final.abs() < 1e-4).any(), "LASSO should produce sparse solution"


class TestFenchelConjugate:
    def test_quadratic(self):
        """f*(y) for f(x) = (1/2) x^T A x + b^T x."""
        A = torch.eye(3) * 2.0
        b = torch.zeros(3)
        f_star = fenchel_conjugate_quadratic(A, b)
        y = torch.tensor([1.0, 0.0, 0.0])
        # f*(y) = (1/2) y^T A^{-1} y = (1/2)(1/2) = 0.25
        result = f_star(y)
        assert abs(result.item() - 0.25) < 1e-5
