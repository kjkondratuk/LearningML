"""
Tests for Module 05: Support Vector Machines

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    sklearn_svm_linear,
    sklearn_svm_rbf,
    compare_kernels,
    visualize_decision_boundary,
    grid_search_svm,
)


@pytest.fixture
def linearly_separable():
    rng = np.random.RandomState(42)
    X = rng.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X[:160], y[:160], X[160:], y[160:]


@pytest.fixture
def nonlinear_data():
    rng = np.random.RandomState(42)
    X = rng.randn(300, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1.0).astype(int)
    return X[:240], y[:240], X[240:], y[240:]


# ---------------------------------------------------------------------------
# sklearn_svm_linear
# ---------------------------------------------------------------------------

class TestSVMLinear:
    def test_output_shape(self, linearly_separable):
        X_train, y_train, X_test, y_test = linearly_separable
        y_pred = sklearn_svm_linear(X_train, y_train, X_test)
        assert y_pred.shape == (40,)

    def test_accuracy(self, linearly_separable):
        X_train, y_train, X_test, y_test = linearly_separable
        y_pred = sklearn_svm_linear(X_train, y_train, X_test)
        accuracy = np.mean(y_pred == y_test)
        assert accuracy > 0.85

    def test_binary_output(self, linearly_separable):
        X_train, y_train, X_test, _ = linearly_separable
        y_pred = sklearn_svm_linear(X_train, y_train, X_test)
        assert set(np.unique(y_pred)).issubset({0, 1})


# ---------------------------------------------------------------------------
# sklearn_svm_rbf
# ---------------------------------------------------------------------------

class TestSVMRBF:
    def test_output_shape(self, nonlinear_data):
        X_train, y_train, X_test, _ = nonlinear_data
        y_pred = sklearn_svm_rbf(X_train, y_train, X_test)
        assert y_pred.shape == X_test.shape[:1]

    def test_handles_nonlinear(self, nonlinear_data):
        X_train, y_train, X_test, y_test = nonlinear_data
        y_pred = sklearn_svm_rbf(X_train, y_train, X_test)
        accuracy = np.mean(y_pred == y_test)
        assert accuracy > 0.75

    def test_custom_params(self, linearly_separable):
        X_train, y_train, X_test, _ = linearly_separable
        y_pred = sklearn_svm_rbf(X_train, y_train, X_test, C=10.0, gamma=0.1)
        assert y_pred.shape == (40,)


# ---------------------------------------------------------------------------
# compare_kernels
# ---------------------------------------------------------------------------

class TestCompareKernels:
    def test_returns_all_kernels(self, linearly_separable):
        X_train, y_train, X_test, y_test = linearly_separable
        result = compare_kernels(X_train, y_train, X_test, y_test)
        assert "linear" in result
        assert "rbf" in result
        assert "poly" in result

    def test_values_are_floats(self, linearly_separable):
        X_train, y_train, X_test, y_test = linearly_separable
        result = compare_kernels(X_train, y_train, X_test, y_test)
        for key, val in result.items():
            assert isinstance(val, float), f"{key} should be float"

    def test_accuracies_reasonable(self, linearly_separable):
        X_train, y_train, X_test, y_test = linearly_separable
        result = compare_kernels(X_train, y_train, X_test, y_test)
        for key, val in result.items():
            assert 0.0 <= val <= 1.0, f"{key} accuracy out of range"
            assert val > 0.5, f"{key} should beat random guessing"


# ---------------------------------------------------------------------------
# visualize_decision_boundary
# ---------------------------------------------------------------------------

class TestVisualizeBoundary:
    def test_output_shapes(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        xx, yy, Z = visualize_decision_boundary(X, y, resolution=0.5)
        assert xx.shape == yy.shape == Z.shape
        assert xx.ndim == 2

    def test_covers_data_range(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        xx, yy, Z = visualize_decision_boundary(X, y, resolution=0.5)
        assert xx.min() <= X[:, 0].min()
        assert xx.max() >= X[:, 0].max()
        assert yy.min() <= X[:, 1].min()
        assert yy.max() >= X[:, 1].max()

    def test_predictions_are_class_labels(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 2)
        y = (X[:, 0] > 0).astype(int)
        _, _, Z = visualize_decision_boundary(X, y, resolution=0.5)
        assert set(np.unique(Z)).issubset({0, 1})


# ---------------------------------------------------------------------------
# grid_search_svm
# ---------------------------------------------------------------------------

class TestGridSearchSVM:
    def test_returns_required_keys(self, linearly_separable):
        X_train, y_train, X_test, y_test = linearly_separable
        result = grid_search_svm(X_train, y_train, X_test, y_test)
        for key in ["best_C", "best_gamma", "best_cv_score", "test_accuracy"]:
            assert key in result

    def test_best_params_in_grid(self, linearly_separable):
        X_train, y_train, X_test, y_test = linearly_separable
        result = grid_search_svm(X_train, y_train, X_test, y_test)
        assert result["best_C"] in [0.1, 1, 10, 100]
        assert result["best_gamma"] in [0.001, 0.01, 0.1, 1]

    def test_cv_score_reasonable(self, linearly_separable):
        X_train, y_train, X_test, y_test = linearly_separable
        result = grid_search_svm(X_train, y_train, X_test, y_test)
        assert result["best_cv_score"] > 0.7
        assert result["test_accuracy"] > 0.7

    def test_scores_are_floats(self, linearly_separable):
        X_train, y_train, X_test, y_test = linearly_separable
        result = grid_search_svm(X_train, y_train, X_test, y_test)
        assert isinstance(result["best_cv_score"], float)
        assert isinstance(result["test_accuracy"], float)
