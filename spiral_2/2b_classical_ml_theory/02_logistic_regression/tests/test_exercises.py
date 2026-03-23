"""
Tests for Logistic Regression exercises.
"""

import numpy as np
import pytest

from exercises import (
    sigmoid,
    log_likelihood_binary,
    gradient_log_likelihood,
    hessian_log_likelihood,
    logistic_regression_gd,
    logistic_regression_newton,
    softmax,
    multinomial_logistic_regression,
    decision_boundary,
)


class TestSigmoid:
    def test_at_zero(self):
        np.testing.assert_allclose(sigmoid(np.array([0.0])), [0.5], atol=1e-10)

    def test_maps_to_unit_interval(self):
        z = np.linspace(-100, 100, 1000)
        s = sigmoid(z)
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_symmetry(self):
        z = np.array([2.0])
        np.testing.assert_allclose(sigmoid(z) + sigmoid(-z), [1.0], atol=1e-10)

    def test_large_values_stable(self):
        """Should not overflow for large inputs."""
        assert sigmoid(np.array([500.0]))[0] < 1.0 + 1e-10
        assert sigmoid(np.array([-500.0]))[0] > -1e-10


class TestLogLikelihood:
    def test_concavity(self):
        """Hessian should be negative semi-definite at random points."""
        np.random.seed(42)
        X = np.random.randn(20, 3)
        y = (np.random.rand(20) > 0.5).astype(float)
        for _ in range(5):
            w = np.random.randn(3)
            H = hessian_log_likelihood(X, y, w)
            eigenvalues = np.linalg.eigvalsh(H)
            assert np.all(eigenvalues <= 1e-10)  # NSD


class TestNewtonVsGD:
    def test_both_converge_to_same(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        w_true = np.array([1.0, -2.0, 0.5])
        p = sigmoid(X @ w_true)
        y = (np.random.rand(100) < p).astype(float)

        w_gd = logistic_regression_gd(X, y, lr=0.01, n_steps=2000)
        w_newton = logistic_regression_newton(X, y, n_steps=20)
        # Both should find similar weights
        np.testing.assert_allclose(w_gd, w_newton, atol=0.5)

    def test_newton_fewer_iterations(self):
        """Newton should get lower LL in fewer steps."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        w_true = np.array([1.0, -2.0, 0.5])
        p = sigmoid(X @ w_true)
        y = (np.random.rand(100) < p).astype(float)

        w_newton = logistic_regression_newton(X, y, n_steps=10)
        w_gd = logistic_regression_gd(X, y, lr=0.01, n_steps=10)
        ll_newton = log_likelihood_binary(X, y, w_newton)
        ll_gd = log_likelihood_binary(X, y, w_gd)
        assert ll_newton >= ll_gd


class TestSoftmax:
    def test_sums_to_one(self):
        z = np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])
        s = softmax(z)
        np.testing.assert_allclose(s.sum(axis=1), [1.0, 1.0], atol=1e-10)

    def test_uniform_for_equal_inputs(self):
        z = np.array([1.0, 1.0, 1.0])
        s = softmax(z.reshape(1, -1))
        np.testing.assert_allclose(s[0], [1/3, 1/3, 1/3], atol=1e-10)

    def test_numerical_stability(self):
        """Should not overflow for large inputs."""
        z = np.array([[1000.0, 1001.0, 999.0]])
        s = softmax(z)
        assert np.all(np.isfinite(s))
        np.testing.assert_allclose(s.sum(), 1.0, atol=1e-10)


class TestMultinomialLR:
    def test_linearly_separable(self):
        """Should achieve high accuracy on easy linearly separable data."""
        np.random.seed(42)
        n = 100
        X = np.vstack([
            np.random.randn(n, 2) + [3, 0],
            np.random.randn(n, 2) + [0, 3],
            np.random.randn(n, 2) + [-3, -3],
        ])
        y = np.array([0] * n + [1] * n + [2] * n)
        W = multinomial_logistic_regression(X, y, n_classes=3, lr=0.01, n_steps=500)
        probs = softmax(X @ W)
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == y)
        assert acc > 0.90


class TestDecisionBoundary:
    def test_output_shapes(self):
        w = np.array([1.0, -1.0])
        x1, x2 = decision_boundary(w, 0.0, (-5, 5))
        assert len(x1) == 100
        assert len(x2) == 100
