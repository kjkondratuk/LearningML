"""
Tests for Dimensionality Reduction exercises.
"""

import numpy as np
import pytest

from exercises import (
    pca_via_eigendecomposition,
    pca_via_svd,
    reconstruct_from_pca,
    pca_explained_variance_plot,
    kernel_pca,
    pca_whitening,
)


class TestPCA:
    def test_eigen_and_svd_agree(self):
        np.random.seed(42)
        X = np.random.randn(50, 10)
        X1, c1, v1 = pca_via_eigendecomposition(X, 3)
        X2, c2, v2 = pca_via_svd(X, 3)
        # Components might differ by sign
        for i in range(3):
            sim = abs(np.dot(c1[i], c2[i]))
            np.testing.assert_allclose(sim, 1.0, atol=1e-6)

    def test_explained_variance_sums_to_one(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        _, _, v = pca_via_eigendecomposition(X, 5)
        np.testing.assert_allclose(v.sum(), 1.0, atol=1e-6)

    def test_reconstruction_error_decreases(self):
        np.random.seed(42)
        X = np.random.randn(50, 10)
        mean = X.mean(axis=0)
        errors = []
        for k in [1, 3, 5, 10]:
            Xt, comp, _ = pca_via_svd(X, k)
            Xr = reconstruct_from_pca(Xt, comp, mean)
            err = np.mean((X - Xr) ** 2)
            errors.append(err)
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] - 1e-10

    def test_pca_2d_max_variance(self):
        """PCA of data along one axis should find that axis."""
        np.random.seed(42)
        X = np.zeros((100, 2))
        X[:, 0] = np.random.randn(100) * 10  # High variance
        X[:, 1] = np.random.randn(100) * 0.1  # Low variance
        _, components, _ = pca_via_svd(X, 1)
        # First component should be approximately [1, 0] or [-1, 0]
        assert abs(abs(components[0, 0]) - 1.0) < 0.1


class TestExplainedVariancePlot:
    def test_cumulative_reaches_one(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        indiv, cumul = pca_explained_variance_plot(X, 5)
        np.testing.assert_allclose(cumul[-1], 1.0, atol=1e-6)

    def test_cumulative_monotonic(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        _, cumul = pca_explained_variance_plot(X, 5)
        for i in range(len(cumul) - 1):
            assert cumul[i + 1] >= cumul[i] - 1e-10


class TestKernelPCA:
    def test_rbf_separates_circles(self):
        """Kernel PCA with RBF should separate concentric circles."""
        np.random.seed(42)
        n = 100
        theta = np.random.uniform(0, 2 * np.pi, n)
        inner = np.column_stack([np.cos(theta) * 1, np.sin(theta) * 1])
        outer = np.column_stack([np.cos(theta) * 3, np.sin(theta) * 3])
        X = np.vstack([inner, outer])
        y = np.array([0] * n + [1] * n)

        def rbf(x1, x2, gamma=1.0):
            return np.exp(-gamma * np.sum((x1 - x2) ** 2))

        Xt = kernel_pca(X, 1, rbf)
        # After kernel PCA, classes should be more separable
        mean0 = Xt[y == 0].mean()
        mean1 = Xt[y == 1].mean()
        assert abs(mean0 - mean1) > 0.1


class TestWhitening:
    def test_identity_covariance(self):
        np.random.seed(42)
        X = np.random.randn(200, 5) @ np.random.randn(5, 5) + 3
        Xw = pca_whitening(X)
        cov = np.cov(Xw.T)
        np.testing.assert_allclose(cov, np.eye(5), atol=0.2)
