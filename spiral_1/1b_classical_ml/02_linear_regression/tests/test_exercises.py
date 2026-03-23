"""
Tests for Module 02: Linear Regression

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    predict_linear,
    compute_loss_mse,
    gradient_descent_step,
    train_linear_regression,
    sklearn_linear_regression,
)


# ---------------------------------------------------------------------------
# predict_linear
# ---------------------------------------------------------------------------

class TestPredictLinear:
    def test_simple(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        w = np.array([0.5, -1.0])
        b = 1.0
        expected = np.array([1.0 * 0.5 + 2.0 * (-1.0) + 1.0,
                             3.0 * 0.5 + 4.0 * (-1.0) + 1.0])
        result = predict_linear(X, w, b)
        np.testing.assert_allclose(result, expected)

    def test_zero_weights(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        w = np.array([0.0, 0.0])
        b = 5.0
        result = predict_linear(X, w, b)
        np.testing.assert_allclose(result, np.array([5.0, 5.0]))

    def test_single_feature(self):
        X = np.array([[2.0], [4.0], [6.0]])
        w = np.array([3.0])
        b = -1.0
        expected = np.array([5.0, 11.0, 17.0])
        np.testing.assert_allclose(predict_linear(X, w, b), expected)

    def test_output_shape(self):
        X = np.ones((10, 3))
        w = np.ones(3)
        result = predict_linear(X, w, 0.0)
        assert result.shape == (10,)


# ---------------------------------------------------------------------------
# compute_loss_mse
# ---------------------------------------------------------------------------

class TestComputeLossMSE:
    def test_zero_loss(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_loss_mse(y, y) == pytest.approx(0.0)

    def test_known_loss(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 5.0])
        assert compute_loss_mse(y_true, y_pred) == pytest.approx(4.0 / 3.0)

    def test_always_non_negative(self):
        rng = np.random.RandomState(42)
        y_true = rng.randn(50)
        y_pred = rng.randn(50)
        assert compute_loss_mse(y_true, y_pred) >= 0.0


# ---------------------------------------------------------------------------
# gradient_descent_step
# ---------------------------------------------------------------------------

class TestGradientDescentStep:
    def test_weights_change(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([1.0, 2.0])
        w = np.zeros(2)
        b = 0.0
        w_new, b_new = gradient_descent_step(X, y, w, b, learning_rate=0.01)
        assert not np.allclose(w_new, w), "Weights should have changed"

    def test_moves_toward_solution(self):
        # Perfect data: y = 2*x + 1
        X = np.array([[1.0], [2.0], [3.0], [4.0]])
        y = np.array([3.0, 5.0, 7.0, 9.0])
        w = np.array([0.0])
        b = 0.0
        lr = 0.01

        loss_before = compute_loss_mse(y, predict_linear(X, w, b))
        for _ in range(100):
            w, b = gradient_descent_step(X, y, w, b, lr)
        loss_after = compute_loss_mse(y, predict_linear(X, w, b))
        assert loss_after < loss_before

    def test_zero_gradient_at_optimum(self):
        # If predictions are perfect, weights should barely change
        X = np.array([[1.0], [2.0]])
        y = np.array([3.0, 5.0])
        w = np.array([2.0])
        b = 1.0
        w_new, b_new = gradient_descent_step(X, y, w, b, learning_rate=0.01)
        np.testing.assert_allclose(w_new, w, atol=1e-10)
        assert abs(b_new - b) < 1e-10


# ---------------------------------------------------------------------------
# train_linear_regression
# ---------------------------------------------------------------------------

class TestTrainLinearRegression:
    def test_learns_simple_function(self):
        # y = 3*x1 + 2*x2 + 1
        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y = 3 * X[:, 0] + 2 * X[:, 1] + 1

        w, b, losses = train_linear_regression(X, y, learning_rate=0.05, n_iterations=500)
        np.testing.assert_allclose(w, [3.0, 2.0], atol=0.1)
        assert abs(b - 1.0) < 0.1

    def test_loss_decreases(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 1)
        y = 2 * X[:, 0] + 3

        _, _, losses = train_linear_regression(X, y, learning_rate=0.05, n_iterations=200)
        assert len(losses) == 200
        assert losses[-1] < losses[0], "Loss should decrease over training"

    def test_returns_correct_types(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([2.0, 4.0, 6.0])
        w, b, losses = train_linear_regression(X, y, n_iterations=10)
        assert isinstance(w, np.ndarray)
        assert isinstance(b, float)
        assert isinstance(losses, list)
        assert w.shape == (1,)


# ---------------------------------------------------------------------------
# sklearn_linear_regression
# ---------------------------------------------------------------------------

class TestSklearnLinearRegression:
    def test_fits_simple(self):
        X_train = np.array([[1.0], [2.0], [3.0], [4.0]])
        y_train = np.array([2.0, 4.0, 6.0, 8.0])
        X_test = np.array([[5.0], [6.0]])

        y_pred, coef, intercept = sklearn_linear_regression(X_train, y_train, X_test)
        np.testing.assert_allclose(y_pred, [10.0, 12.0], atol=0.1)
        np.testing.assert_allclose(coef, [2.0], atol=0.1)
        assert abs(intercept - 0.0) < 0.1

    def test_output_shapes(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 3)
        y_train = rng.randn(50)
        X_test = rng.randn(10, 3)

        y_pred, coef, intercept = sklearn_linear_regression(X_train, y_train, X_test)
        assert y_pred.shape == (10,)
        assert coef.shape == (3,)
        assert isinstance(intercept, float)

    def test_intercept_is_float(self):
        X_train = np.array([[1.0], [2.0]])
        y_train = np.array([1.0, 2.0])
        X_test = np.array([[3.0]])
        _, _, intercept = sklearn_linear_regression(X_train, y_train, X_test)
        assert isinstance(intercept, float)
