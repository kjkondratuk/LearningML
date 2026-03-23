"""
Tests for Module 03: Logistic Regression

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    sigmoid,
    predict_proba,
    predict_class,
    compute_log_loss,
    train_logistic_regression,
    sklearn_logistic_regression,
)


# ---------------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------------

class TestSigmoid:
    def test_zero(self):
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)

    def test_large_positive(self):
        result = sigmoid(np.array([100.0]))[0]
        assert result == pytest.approx(1.0, abs=1e-10)

    def test_large_negative(self):
        result = sigmoid(np.array([-100.0]))[0]
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self):
        z = np.array([2.0])
        assert sigmoid(z)[0] + sigmoid(-z)[0] == pytest.approx(1.0)

    def test_output_range(self):
        z = np.linspace(-10, 10, 100)
        result = sigmoid(z)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_monotonic(self):
        z = np.linspace(-5, 5, 50)
        result = sigmoid(z)
        assert np.all(np.diff(result) > 0), "Sigmoid should be monotonically increasing"

    def test_no_overflow(self):
        """Sigmoid should handle extreme values without overflow warnings."""
        z = np.array([500.0, -500.0, 1000.0, -1000.0])
        result = sigmoid(z)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


# ---------------------------------------------------------------------------
# predict_proba
# ---------------------------------------------------------------------------

class TestPredictProba:
    def test_output_range(self):
        X = np.random.RandomState(0).randn(20, 3)
        w = np.array([1.0, -0.5, 0.3])
        b = 0.1
        proba = predict_proba(X, w, b)
        assert np.all(proba > 0)
        assert np.all(proba < 1)

    def test_shape(self):
        X = np.ones((10, 2))
        w = np.ones(2)
        proba = predict_proba(X, w, 0.0)
        assert proba.shape == (10,)

    def test_zero_weights(self):
        X = np.random.RandomState(1).randn(5, 2)
        w = np.zeros(2)
        b = 0.0
        proba = predict_proba(X, w, b)
        np.testing.assert_allclose(proba, 0.5)


# ---------------------------------------------------------------------------
# predict_class
# ---------------------------------------------------------------------------

class TestPredictClass:
    def test_default_threshold(self):
        X = np.array([[10.0], [-10.0]])
        w = np.array([1.0])
        b = 0.0
        preds = predict_class(X, w, b)
        np.testing.assert_array_equal(preds, [1, 0])

    def test_custom_threshold(self):
        X = np.array([[0.0]])  # sigmoid(0) = 0.5
        w = np.array([1.0])
        b = 0.0
        assert predict_class(X, w, b, threshold=0.5)[0] == 1
        assert predict_class(X, w, b, threshold=0.6)[0] == 0

    def test_output_values(self):
        X = np.random.RandomState(42).randn(50, 2)
        w = np.array([1.0, -1.0])
        preds = predict_class(X, w, 0.0)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_output_shape(self):
        X = np.ones((7, 3))
        w = np.ones(3)
        assert predict_class(X, w, 0.0).shape == (7,)


# ---------------------------------------------------------------------------
# compute_log_loss
# ---------------------------------------------------------------------------

class TestLogLoss:
    def test_perfect_prediction(self):
        y_true = np.array([1, 0, 1])
        y_proba = np.array([0.999, 0.001, 0.999])
        loss = compute_log_loss(y_true, y_proba)
        assert loss < 0.01

    def test_worst_prediction(self):
        y_true = np.array([1, 0])
        y_proba = np.array([0.001, 0.999])
        loss = compute_log_loss(y_true, y_proba)
        assert loss > 5.0

    def test_coin_flip(self):
        y_true = np.array([1, 0, 1, 0])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5])
        expected = -np.log(0.5)  # ln(2) ~ 0.693
        assert compute_log_loss(y_true, y_proba) == pytest.approx(expected, abs=1e-6)

    def test_non_negative(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        y_proba = rng.uniform(0.01, 0.99, 100)
        assert compute_log_loss(y_true, y_proba) >= 0.0

    def test_handles_extreme_probabilities(self):
        """Should not produce inf or nan with probabilities very close to 0 or 1."""
        y_true = np.array([1, 0])
        y_proba = np.array([1e-15, 1 - 1e-15])
        loss = compute_log_loss(y_true, y_proba)
        assert np.isfinite(loss)


# ---------------------------------------------------------------------------
# train_logistic_regression
# ---------------------------------------------------------------------------

class TestTrainLogisticRegression:
    def test_learns_linearly_separable(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(float)

        w, b, losses = train_logistic_regression(
            X, y, learning_rate=0.5, n_iterations=500
        )

        # Model should classify most correctly
        proba = 1 / (1 + np.exp(-(X @ w + b)))
        preds = (proba >= 0.5).astype(float)
        accuracy = np.mean(preds == y)
        assert accuracy > 0.9

    def test_loss_decreases(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 2)
        y = (X[:, 0] > 0).astype(float)

        _, _, losses = train_logistic_regression(
            X, y, learning_rate=0.1, n_iterations=200
        )
        assert losses[-1] < losses[0]

    def test_returns_correct_types(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0.0, 1.0, 1.0])
        w, b, losses = train_logistic_regression(X, y, n_iterations=10)
        assert isinstance(w, np.ndarray)
        assert isinstance(b, float)
        assert isinstance(losses, list)
        assert w.shape == (2,)
        assert len(losses) == 10


# ---------------------------------------------------------------------------
# sklearn_logistic_regression
# ---------------------------------------------------------------------------

class TestSklearnLogisticRegression:
    def test_separable_data(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 2)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(float)
        X_test = rng.randn(20, 2)
        y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(float)

        y_pred, y_proba = sklearn_logistic_regression(X_train, y_train, X_test)
        accuracy = np.mean(y_pred == y_test)
        assert accuracy > 0.85

    def test_output_shapes(self):
        rng = np.random.RandomState(0)
        X_train = rng.randn(50, 3)
        y_train = rng.randint(0, 2, 50).astype(float)
        X_test = rng.randn(10, 3)

        y_pred, y_proba = sklearn_logistic_regression(X_train, y_train, X_test)
        assert y_pred.shape == (10,)
        assert y_proba.shape == (10,)

    def test_proba_range(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 2)
        y_train = rng.randint(0, 2, 50).astype(float)
        X_test = rng.randn(10, 2)

        _, y_proba = sklearn_logistic_regression(X_train, y_train, X_test)
        assert np.all(y_proba >= 0)
        assert np.all(y_proba <= 1)
