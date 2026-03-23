"""
Tests for Hyperparameter Tuning exercises.
"""

import numpy as np
import pytest

from exercises import (
    grid_search,
    gaussian_process_surrogate,
    expected_improvement,
    bayesian_optimization,
)


class TestGridSearch:
    def test_evaluates_all_combinations(self):
        call_count = [0]
        def model_fn(X_tr, y_tr, X_te, alpha=1.0):
            call_count[0] += 1
            return -np.mean((y_tr - np.mean(y_tr)) ** 2)

        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        grid = {"alpha": [0.1, 1.0, 10.0]}
        best, results = grid_search(model_fn, grid, X, y, k_folds=3)
        assert len(results) == 3  # 3 alpha values


class TestGPSurrogate:
    def test_uncertainty_increases_far_from_data(self):
        X_obs = np.array([[0.0], [1.0], [2.0]])
        y_obs = np.array([0.0, 1.0, 4.0])
        X_near = np.array([[1.5]])
        X_far = np.array([[10.0]])
        _, sigma_near = gaussian_process_surrogate(X_obs, y_obs, X_near)
        _, sigma_far = gaussian_process_surrogate(X_obs, y_obs, X_far)
        assert sigma_far[0] > sigma_near[0]

    def test_low_uncertainty_at_observations(self):
        X_obs = np.array([[0.0], [1.0], [2.0]])
        y_obs = np.array([0.0, 1.0, 4.0])
        mu, sigma = gaussian_process_surrogate(X_obs, y_obs, X_obs)
        np.testing.assert_allclose(sigma, 0.0, atol=0.01)
        np.testing.assert_allclose(mu, y_obs, atol=0.01)


class TestExpectedImprovement:
    def test_zero_when_no_improvement(self):
        """EI = 0 when mean is worse than best and sigma = 0."""
        mu = np.array([0.5])  # worse than best
        sigma = np.array([0.0])
        y_best = 1.0
        ei = expected_improvement(mu, sigma, y_best)
        np.testing.assert_allclose(ei, 0.0, atol=1e-10)

    def test_positive_with_uncertainty(self):
        mu = np.array([0.5])
        sigma = np.array([1.0])
        y_best = 1.0
        ei = expected_improvement(mu, sigma, y_best)
        assert ei[0] > 0


class TestBayesianOptimization:
    def test_finds_minimum_of_quadratic(self):
        """BO should find the min of f(x) = (x-3)^2."""
        def objective(x):
            return float((x[0] - 3) ** 2)

        best_x, best_y, history = bayesian_optimization(
            objective, {"x": (0.0, 6.0)},
            n_initial=3, n_iterations=15, seed=42
        )
        np.testing.assert_allclose(best_x[0], 3.0, atol=0.5)
