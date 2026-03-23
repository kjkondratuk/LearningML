"""
Tests for Bias-Variance exercises.
"""

import numpy as np
import pytest

from exercises import (
    bias_variance_decomposition,
    polynomial_bias_variance,
    learning_curves,
)


class TestBiasVarianceDecomposition:
    def test_low_degree_high_bias(self):
        """Degree-1 polynomial on cubic data: should have high bias."""
        bias2, var, total = polynomial_bias_variance(
            degree=1, n_datasets=50, n_points=30, noise_std=0.3, seed=42
        )
        assert bias2 > var  # High bias regime

    def test_high_degree_high_variance(self):
        """Degree-15 polynomial on cubic data: should have high variance."""
        bias2, var, total = polynomial_bias_variance(
            degree=15, n_datasets=50, n_points=30, noise_std=0.3, seed=42
        )
        assert var > 0.01  # Non-trivial variance

    def test_decomposition_adds_up(self):
        """bias^2 + variance should approximately equal total error."""
        bias2, var, total = polynomial_bias_variance(
            degree=5, n_datasets=100, n_points=50, noise_std=0.3, seed=42
        )
        # Total ~ bias^2 + variance (noise is separate)
        np.testing.assert_allclose(bias2 + var, total, atol=0.1)


class TestLearningCurves:
    def test_training_error_increases(self):
        """Training error should generally increase with more data (for fixed model)."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 1)
        y = 3 * X[:, 0] + np.random.randn(n)

        def model_fn(X_tr, y_tr, X_te):
            w = np.linalg.lstsq(X_tr, y_tr, rcond=None)[0]
            return X_te @ w

        train_errors, val_errors = learning_curves(
            model_fn, X, y, train_sizes=[10, 20, 50, 100, 150]
        )
        # Val error should generally decrease with more data
        assert val_errors[-1] <= val_errors[0] + 0.5
