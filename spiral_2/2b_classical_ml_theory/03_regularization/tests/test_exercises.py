"""
Tests for Regularization exercises.
"""

import numpy as np
import pytest

from exercises import (
    ridge_regression,
    lasso_coordinate_descent,
    elastic_net,
    regularization_path,
    l1_sparsity_demo,
    cross_validate_alpha,
)


class TestRidge:
    def test_approaches_ols(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1, 2, 3]) + 0.1 * np.random.randn(50)
        w_ridge = ridge_regression(X, y, alpha=1e-10)
        w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
        np.testing.assert_allclose(w_ridge, w_ols, atol=1e-3)

    def test_shrinks_weights(self):
        np.random.seed(42)
        X = np.random.randn(50, 3)
        y = X @ np.array([1, 2, 3]) + 0.1 * np.random.randn(50)
        w_low = ridge_regression(X, y, alpha=0.01)
        w_high = ridge_regression(X, y, alpha=100.0)
        assert np.linalg.norm(w_high) < np.linalg.norm(w_low)


class TestLasso:
    def test_produces_sparse_weights(self):
        np.random.seed(42)
        X = np.random.randn(100, 10)
        w_true = np.array([3, 0, 0, 2, 0, 0, 0, -1, 0, 0], dtype=float)
        y = X @ w_true + 0.1 * np.random.randn(100)
        w = lasso_coordinate_descent(X, y, alpha=0.5, n_steps=100)
        n_zero = np.sum(np.abs(w) < 1e-5)
        assert n_zero >= 5  # Most noise features should be zeroed


class TestElasticNet:
    def test_l1_ratio_0_matches_ridge(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        w_en = elastic_net(X, y, alpha=1.0, l1_ratio=0.0, n_steps=200)
        w_ridge = ridge_regression(X, y, alpha=0.5)  # adjust for formulation
        # Should be similar (not exact due to solver differences)
        assert np.linalg.norm(w_en - w_ridge) < 1.0

    def test_l1_ratio_1_gives_sparsity(self):
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = X[:, 0] + 0.1 * np.random.randn(50)
        w = elastic_net(X, y, alpha=0.5, l1_ratio=1.0, n_steps=200)
        n_zero = np.sum(np.abs(w) < 1e-5)
        assert n_zero >= 5


class TestSparsityDemo:
    def test_recovers_sparse_features(self):
        np.random.seed(42)
        d_true, d_noise = 5, 95
        d = d_true + d_noise
        X = np.random.randn(200, d)
        w_true = np.zeros(d)
        w_true[:d_true] = np.array([3, -2, 1.5, -1, 0.5])
        y = X @ w_true + 0.5 * np.random.randn(200)
        w, n_nonzero = l1_sparsity_demo(X, y, alpha=0.3)
        # Lasso should zero out at least 80% of the noise features
        noise_zeros = np.sum(np.abs(w[d_true:]) < 1e-5)
        assert noise_zeros >= int(0.8 * d_noise)


class TestCrossValidateAlpha:
    def test_selects_reasonable_alpha(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X @ np.array([1, 2, 0, 0, 0]) + 0.5 * np.random.randn(100)
        alphas = np.logspace(-3, 3, 20)
        best_alpha, cv_errors = cross_validate_alpha(
            X, y, alphas, k_folds=5, model_fn=ridge_regression
        )
        assert 1e-3 < best_alpha < 100
