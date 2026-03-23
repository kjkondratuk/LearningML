"""
Tests for SVM exercises.
"""

import numpy as np
import pytest

from exercises import (
    kernel_function,
    smo_simplified,
    predict_svm,
)


class TestKernelFunction:
    def test_linear(self):
        x1 = np.array([1.0, 2.0])
        x2 = np.array([3.0, 4.0])
        assert kernel_function(x1, x2, "linear") == pytest.approx(11.0)

    def test_rbf_same_point(self):
        x = np.array([1.0, 2.0])
        assert kernel_function(x, x, "rbf", gamma=1.0) == pytest.approx(1.0)

    def test_rbf_far_points(self):
        x1 = np.array([0.0, 0.0])
        x2 = np.array([100.0, 100.0])
        val = kernel_function(x1, x2, "rbf", gamma=1.0)
        assert val < 1e-10  # Very far apart


class TestSMO:
    def test_linearly_separable(self):
        np.random.seed(42)
        n = 50
        X = np.vstack([
            np.random.randn(n, 2) + [2, 2],
            np.random.randn(n, 2) + [-2, -2],
        ])
        y = np.array([1] * n + [-1] * n, dtype=float)
        kern = lambda x1, x2: kernel_function(x1, x2, "linear")
        alphas, b = smo_simplified(X, y, C=1.0, kernel_fn=kern)

        # Support vectors are points with alpha > 0
        sv_idx = alphas > 1e-5
        preds = predict_svm(
            X, X[sv_idx], y[sv_idx], alphas[sv_idx], b, kern
        )
        acc = np.mean(preds == y)
        assert acc > 0.95

    def test_kernel_svm_xor(self):
        """RBF kernel SVM should handle XOR-like data."""
        np.random.seed(42)
        X = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]], dtype=float)
        X = X + 0.1 * np.random.randn(4, 2)
        y = np.array([1, 1, -1, -1], dtype=float)
        kern = lambda x1, x2: kernel_function(x1, x2, "rbf", gamma=1.0)
        alphas, b = smo_simplified(X, y, C=10.0, kernel_fn=kern, max_passes=200)
        sv_idx = alphas > 1e-5
        if sv_idx.sum() > 0:
            preds = predict_svm(
                X, X[sv_idx], y[sv_idx], alphas[sv_idx], b, kern
            )
            acc = np.mean(preds == y)
            assert acc >= 0.75


class TestSVMSupport:
    def test_support_vectors_closest_to_boundary(self):
        """Support vectors should be the points closest to the decision boundary."""
        np.random.seed(42)
        n = 50
        X = np.vstack([
            np.random.randn(n, 2) + [3, 0],
            np.random.randn(n, 2) + [-3, 0],
        ])
        y = np.array([1] * n + [-1] * n, dtype=float)
        kern = lambda x1, x2: kernel_function(x1, x2, "linear")
        alphas, b = smo_simplified(X, y, C=1.0, kernel_fn=kern)
        sv_idx = np.where(alphas > 1e-5)[0]

        # Support vectors should exist
        assert len(sv_idx) > 0
        # They should be closer to the boundary than non-support vectors
        distances = np.abs(X @ np.array([1.0, 0.0]) + b)  # approximate
        if len(sv_idx) > 0 and len(sv_idx) < len(y):
            non_sv_idx = np.where(alphas <= 1e-5)[0]
            if len(non_sv_idx) > 0:
                assert distances[sv_idx].mean() <= distances[non_sv_idx].mean() + 1.0
