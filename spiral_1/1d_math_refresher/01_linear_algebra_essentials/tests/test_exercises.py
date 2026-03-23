"""
Tests for Module 01: Linear Algebra Essentials

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    dot_product,
    matrix_multiply,
    linear_transform_2d,
    compute_covariance_matrix,
    eigen_decomposition,
    project_onto_principal_components,
)


# ---------------------------------------------------------------------------
# dot_product
# ---------------------------------------------------------------------------

class TestDotProduct:
    def test_simple(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, 5.0, 6.0])
        assert dot_product(a, b) == pytest.approx(32.0)

    def test_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert dot_product(a, b) == pytest.approx(0.0)

    def test_parallel(self):
        a = np.array([2.0, 3.0])
        b = np.array([4.0, 6.0])
        assert dot_product(a, b) == pytest.approx(26.0)

    def test_matches_numpy(self):
        np.random.seed(42)
        a = np.random.randn(100)
        b = np.random.randn(100)
        assert dot_product(a, b) == pytest.approx(np.dot(a, b), abs=1e-10)


# ---------------------------------------------------------------------------
# matrix_multiply
# ---------------------------------------------------------------------------

class TestMatrixMultiply:
    def test_identity(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        I = np.eye(2)
        result = matrix_multiply(A, I)
        np.testing.assert_allclose(result, A)

    def test_known_product(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        expected = np.array([[19, 22], [43, 50]], dtype=float)
        result = matrix_multiply(A, B)
        np.testing.assert_allclose(result, expected)

    def test_non_square(self):
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 2)
        result = matrix_multiply(A, B)
        expected = A @ B
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_output_shape(self):
        A = np.random.randn(5, 3)
        B = np.random.randn(3, 7)
        result = matrix_multiply(A, B)
        assert result.shape == (5, 7)


# ---------------------------------------------------------------------------
# linear_transform_2d
# ---------------------------------------------------------------------------

class TestLinearTransform2d:
    def test_identity_transform(self):
        A = np.eye(2)
        pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        result = linear_transform_2d(A, pts)
        np.testing.assert_allclose(result, pts)

    def test_rotation_90(self):
        """90-degree counter-clockwise rotation."""
        A = np.array([[0, -1], [1, 0]], dtype=float)
        pts = np.array([[1], [0]], dtype=float)
        result = linear_transform_2d(A, pts)
        np.testing.assert_allclose(result, np.array([[0], [1]]), atol=1e-10)

    def test_scaling(self):
        A = np.array([[2, 0], [0, 3]], dtype=float)
        pts = np.array([[1, 2], [1, 2]], dtype=float)
        result = linear_transform_2d(A, pts)
        expected = np.array([[2, 4], [3, 6]], dtype=float)
        np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------------------
# compute_covariance_matrix
# ---------------------------------------------------------------------------

class TestComputeCovarianceMatrix:
    def test_shape(self):
        X = np.random.randn(100, 3)
        C = compute_covariance_matrix(X)
        assert C.shape == (3, 3)

    def test_symmetric(self):
        X = np.random.randn(50, 4)
        C = compute_covariance_matrix(X)
        np.testing.assert_allclose(C, C.T, atol=1e-10)

    def test_matches_numpy(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        C = compute_covariance_matrix(X)
        expected = np.cov(X, rowvar=False)
        np.testing.assert_allclose(C, expected, atol=1e-10)

    def test_uncorrelated_features(self):
        """Diagonal covariance for independent features."""
        np.random.seed(0)
        X = np.random.randn(10000, 2)
        C = compute_covariance_matrix(X)
        # Off-diagonal should be near zero
        assert abs(C[0, 1]) < 0.1


# ---------------------------------------------------------------------------
# eigen_decomposition
# ---------------------------------------------------------------------------

class TestEigenDecomposition:
    def test_descending_order(self):
        A = np.array([[2, 1], [1, 3]], dtype=float)
        vals, vecs = eigen_decomposition(A)
        assert vals[0] >= vals[1]

    def test_eigenvector_equation(self):
        """A @ v = lambda * v for each eigenpair."""
        A = np.array([[4, 1], [1, 3]], dtype=float)
        vals, vecs = eigen_decomposition(A)
        for i in range(len(vals)):
            lhs = A @ vecs[:, i]
            rhs = vals[i] * vecs[:, i]
            np.testing.assert_allclose(lhs, rhs, atol=1e-10)

    def test_identity_matrix(self):
        vals, vecs = eigen_decomposition(np.eye(3))
        np.testing.assert_allclose(vals, np.ones(3), atol=1e-10)


# ---------------------------------------------------------------------------
# project_onto_principal_components
# ---------------------------------------------------------------------------

class TestProjectOntoPrincipalComponents:
    def test_output_shape(self):
        X = np.random.randn(50, 5)
        proj, _ = project_onto_principal_components(X, n_components=2)
        assert proj.shape == (50, 2)

    def test_explained_variance_ratio(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        _, ratio = project_onto_principal_components(X, n_components=3)
        assert ratio == pytest.approx(1.0, abs=1e-10)

    def test_partial_variance(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        _, ratio = project_onto_principal_components(X, n_components=2)
        assert 0.0 < ratio < 1.0

    def test_correlated_data(self):
        """With strongly correlated features, 1 component should explain most."""
        np.random.seed(42)
        base = np.random.randn(200, 1)
        X = np.hstack([base, base + 0.01 * np.random.randn(200, 1)])
        _, ratio = project_onto_principal_components(X, n_components=1)
        assert ratio > 0.99
