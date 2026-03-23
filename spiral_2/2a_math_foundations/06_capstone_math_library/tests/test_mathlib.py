"""
Tests for the mathlib capstone library.

Must all pass in under 30 seconds.
"""

import numpy as np
import pytest

from mathlib import (
    lu_decomposition,
    qr_decomposition,
    eigendecomposition,
    svd,
    pseudoinverse,
    solve,
    is_positive_definite,
    gradient,
    jacobian,
    hessian,
    gaussian_pdf,
    mle_gaussian,
    map_gaussian,
    gradient_descent,
    adam_optimizer,
    entropy,
    kl_divergence,
    cross_entropy,
)


# ── Linear Algebra ────────────────────────────────────────────────────────────

class TestLinearAlgebra:
    def test_lu_reconstruction(self):
        np.random.seed(42)
        A = np.random.randn(4, 4)
        L, U = lu_decomposition(A)
        np.testing.assert_allclose(L @ U, A, atol=1e-10)

    def test_qr_reconstruction(self):
        np.random.seed(42)
        A = np.random.randn(5, 3)
        Q, R = qr_decomposition(A)
        np.testing.assert_allclose(Q @ R, A, atol=1e-10)
        np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=1e-10)

    def test_eigendecomposition(self):
        np.random.seed(42)
        M = np.random.randn(3, 3)
        A = M + M.T
        vals, vecs = eigendecomposition(A)
        np_vals = np.sort(np.linalg.eigvalsh(A))
        np.testing.assert_allclose(np.sort(vals), np_vals, atol=1e-4)

    def test_svd_reconstruction(self):
        np.random.seed(42)
        A = np.random.randn(4, 3)
        U, S, Vt = svd(A)
        k = len(S)
        recon = U[:, :k] @ np.diag(S) @ Vt[:k, :]
        np.testing.assert_allclose(recon, A, atol=1e-6)

    def test_pseudoinverse_least_squares(self):
        np.random.seed(42)
        A = np.random.randn(5, 3)
        b = np.random.randn(5)
        x = pseudoinverse(A) @ b
        x_np = np.linalg.lstsq(A, b, rcond=None)[0]
        np.testing.assert_allclose(x, x_np, atol=1e-6)

    def test_solve(self):
        np.random.seed(42)
        A = np.random.randn(4, 4)
        b = np.random.randn(4)
        x = solve(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-8)

    def test_positive_definite(self):
        M = np.random.randn(3, 3)
        assert is_positive_definite(M @ M.T + 0.1 * np.eye(3)) is True
        assert is_positive_definite(-np.eye(3)) is False


# ── Calculus ──────────────────────────────────────────────────────────────────

class TestCalculus:
    def test_gradient(self):
        f = lambda x: float(x[0] ** 2 + x[1] ** 2)
        g = gradient(f, np.array([3.0, 4.0]))
        np.testing.assert_allclose(g, [6.0, 8.0], atol=1e-4)

    def test_jacobian(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        f = lambda x: A @ x
        J = jacobian(f, np.array([1.0, 1.0]))
        np.testing.assert_allclose(J, A, atol=1e-4)

    def test_hessian(self):
        A = np.array([[4, 1], [1, 3]], dtype=float)
        f = lambda x: float(0.5 * x @ A @ x)
        H = hessian(f, np.array([1.0, 2.0]))
        np.testing.assert_allclose(H, A, atol=1e-3)


# ── Probability ───────────────────────────────────────────────────────────────

class TestProbability:
    def test_gaussian_pdf_at_mean(self):
        val = gaussian_pdf(np.array([0.0]), 0.0, 1.0)
        np.testing.assert_allclose(val, [1.0 / np.sqrt(2 * np.pi)], atol=1e-10)

    def test_mle(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu, var = mle_gaussian(data)
        np.testing.assert_allclose(mu, 3.0, atol=1e-10)
        np.testing.assert_allclose(var, 2.0, atol=1e-10)

    def test_map_shrinkage(self):
        data = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        mu_map = map_gaussian(data, 0.0, 1.0, 2.0)
        assert 0.0 < mu_map < np.mean(data)


# ── Optimization ──────────────────────────────────────────────────────────────

class TestOptimization:
    def test_gd_converges(self):
        grad_f = lambda x: 2 * x
        traj = gradient_descent(grad_f, np.array([5.0, 5.0]), lr=0.1, n_steps=100)
        np.testing.assert_allclose(traj[-1], [0.0, 0.0], atol=1e-3)

    def test_adam_converges(self):
        grad_f = lambda x: 2 * x
        traj = adam_optimizer(grad_f, np.array([5.0, 5.0]), lr=0.1, n_steps=200)
        np.testing.assert_allclose(traj[-1], [0.0, 0.0], atol=1e-2)


# ── Information Theory ────────────────────────────────────────────────────────

class TestInfoTheory:
    def test_entropy_fair_coin(self):
        np.testing.assert_allclose(entropy(np.array([0.5, 0.5])), 1.0, atol=1e-10)

    def test_kl_zero_same(self):
        p = np.array([0.3, 0.4, 0.3])
        np.testing.assert_allclose(kl_divergence(p, p), 0.0, atol=1e-10)

    def test_cross_entropy_equals_entropy(self):
        p = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(cross_entropy(p, p), entropy(p), atol=1e-10)


# ── Integration Test: Linear Regression via mathlib ───────────────────────────

class TestLinearRegressionEndToEnd:
    """Solve linear regression using ONLY mathlib functions."""

    def test_solve_regression_via_pseudoinverse(self):
        np.random.seed(42)
        n, d = 50, 3
        X = np.random.randn(n, d)
        w_true = np.array([1.5, -2.0, 0.5])
        y = X @ w_true + 0.1 * np.random.randn(n)

        # Solve using pseudoinverse from mathlib
        w_hat = pseudoinverse(X) @ y
        np.testing.assert_allclose(w_hat, w_true, atol=0.1)

    def test_solve_regression_via_normal_equations(self):
        np.random.seed(42)
        n, d = 50, 3
        X = np.random.randn(n, d)
        w_true = np.array([1.5, -2.0, 0.5])
        y = X @ w_true + 0.1 * np.random.randn(n)

        # Normal equations: w = (X^T X)^{-1} X^T y
        # Solve X^T X w = X^T y using our solve function
        w_hat = solve(X.T @ X, X.T @ y)
        np.testing.assert_allclose(w_hat, w_true, atol=0.1)
