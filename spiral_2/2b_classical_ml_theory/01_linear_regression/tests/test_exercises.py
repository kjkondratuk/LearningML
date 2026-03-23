"""
Tests for Linear Regression exercises.
"""

import numpy as np
import pytest

from exercises import (
    ols_closed_form,
    ols_via_gradient_descent,
    ols_via_svd,
    mle_linear_regression,
    map_linear_regression,
    bayesian_linear_regression,
    predictive_distribution,
    polynomial_regression,
)


@pytest.fixture
def simple_regression_data():
    np.random.seed(42)
    n, d = 50, 3
    X = np.random.randn(n, d)
    w_true = np.array([1.5, -2.0, 0.5])
    y = X @ w_true + 0.1 * np.random.randn(n)
    return X, y, w_true


class TestOLSClosedForm:
    def test_recovers_weights(self, simple_regression_data):
        X, y, w_true = simple_regression_data
        w = ols_closed_form(X, y)
        np.testing.assert_allclose(w, w_true, atol=0.1)


class TestOLSMethods:
    """OLS closed form, GD, and SVD all produce same weights."""

    def test_all_methods_agree(self, simple_regression_data):
        X, y, w_true = simple_regression_data
        w_cf = ols_closed_form(X, y)
        w_gd = ols_via_gradient_descent(X, y, lr=0.01, n_steps=5000)
        w_svd = ols_via_svd(X, y)
        np.testing.assert_allclose(w_gd, w_cf, atol=0.1)
        np.testing.assert_allclose(w_svd, w_cf, atol=1e-6)


class TestMLEMatchesOLS:
    def test_mle_equals_ols(self, simple_regression_data):
        X, y, _ = simple_regression_data
        w_ols = ols_closed_form(X, y)
        w_mle = mle_linear_regression(X, y)
        np.testing.assert_allclose(w_mle, w_ols, atol=1e-10)


class TestMAPRegression:
    def test_approaches_ols_as_precision_zero(self, simple_regression_data):
        X, y, _ = simple_regression_data
        w_ols = ols_closed_form(X, y)
        w_map = map_linear_regression(X, y, prior_precision=1e-10)
        np.testing.assert_allclose(w_map, w_ols, atol=1e-3)

    def test_shrinks_toward_zero(self, simple_regression_data):
        X, y, _ = simple_regression_data
        w_map = map_linear_regression(X, y, prior_precision=1000.0)
        # With very strong prior, weights should be near zero
        assert np.linalg.norm(w_map) < 0.1


class TestBayesianRegression:
    def test_uncertainty_increases_away_from_data(self):
        np.random.seed(42)
        X = np.random.randn(20, 1) * 2
        y = 3 * X[:, 0] + 0.5 * np.random.randn(20)

        mu, cov = bayesian_linear_regression(X, y, 1.0, 4.0)
        X_near = np.array([[0.0]])
        X_far = np.array([[10.0]])
        _, var_near = predictive_distribution(X_near, mu, cov, 4.0)
        _, var_far = predictive_distribution(X_far, mu, cov, 4.0)
        assert var_far[0] > var_near[0]


class TestPolynomialRegression:
    def test_degree_1_matches_ols(self):
        np.random.seed(42)
        X = np.random.randn(30, 1)
        y = 2 * X[:, 0] + 1 + 0.1 * np.random.randn(30)
        w = polynomial_regression(X, y, degree=1)
        assert w.shape == (2,)
        np.testing.assert_allclose(w[1], 2.0, atol=0.2)
        np.testing.assert_allclose(w[0], 1.0, atol=0.2)
