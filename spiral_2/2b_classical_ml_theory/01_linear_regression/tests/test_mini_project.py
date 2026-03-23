"""
Tests for the Bayesian Linear Regression mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    generate_synthetic_data,
    sequential_bayesian_update,
    bootstrap_uncertainty,
)


class TestGenerateData:
    def test_correct_shapes(self):
        X, y = generate_synthetic_data(100, np.array([1.0, 2.0]), 0.1)
        assert X.shape == (100, 2)
        assert y.shape == (100,)


class TestSequentialUpdate:
    def test_posterior_sharpens(self):
        X, y = generate_synthetic_data(100, np.array([1.0, -1.0]), 0.5)
        posteriors = sequential_bayesian_update(
            X, y, [5, 10, 50, 100], prior_precision=1.0, noise_precision=4.0
        )
        # Posterior covariance trace should decrease with more data
        traces = [np.trace(cov) for _, cov in posteriors]
        for i in range(len(traces) - 1):
            assert traces[i + 1] <= traces[i] + 1e-10


class TestBootstrap:
    def test_output_shapes(self):
        np.random.seed(42)
        X = np.random.randn(50, 2)
        y = X @ np.array([1.0, -1.0]) + 0.1 * np.random.randn(50)
        X_test = np.random.randn(10, 2)
        mean, std = bootstrap_uncertainty(X, y, X_test, n_bootstrap=100)
        assert mean.shape == (10,)
        assert std.shape == (10,)
        assert np.all(std >= 0)
