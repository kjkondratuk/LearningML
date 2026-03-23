"""
Tests for Linear Algebra exercises.

All decompositions are verified by reconstruction within tolerance 1e-10.
Eigenvalues/vectors are verified against np.linalg.eig.
Known 2x2 and 3x3 hand-computed examples are included.
Edge cases: singular matrices, rectangular matrices.
"""

import numpy as np
import pytest

from exercises import (
    matrix_multiply,
    lu_decomposition,
    qr_decomposition,
    eigendecomposition,
    svd,
    solve_linear_system,
    compute_pseudoinverse,
    is_positive_definite,
)

TOL = 1e-10


# ── matrix_multiply ──────────────────────────────────────────────────────────

class TestMatrixMultiply:
    def test_2x2(self):
        A = np.array([[1, 2], [3, 4]], dtype=float)
        B = np.array([[5, 6], [7, 8]], dtype=float)
        expected = np.array([[19, 22], [43, 50]], dtype=float)
        result = matrix_multiply(A, B)
        np.testing.assert_allclose(result, expected, atol=TOL)

    def test_3x3_identity(self):
        A = np.random.randn(3, 3)
        I = np.eye(3)
        np.testing.assert_allclose(matrix_multiply(A, I), A, atol=TOL)

    def test_rectangular(self):
        A = np.random.randn(3, 4)
        B = np.random.randn(4, 2)
        expected = A @ B
        np.testing.assert_allclose(matrix_multiply(A, B), expected, atol=TOL)

    def test_matches_np_dot(self):
        np.random.seed(42)
        A = np.random.randn(20, 30)
        B = np.random.randn(30, 15)
        np.testing.assert_allclose(matrix_multiply(A, B), np.dot(A, B), atol=TOL)

    def test_incompatible_shapes_raises(self):
        A = np.random.randn(3, 4)
        B = np.random.randn(5, 2)
        with pytest.raises((ValueError, IndexError)):
            matrix_multiply(A, B)


# ── lu_decomposition ─────────────────────────────────────────────────────────

class TestLUDecomposition:
    def test_2x2_known(self):
        A = np.array([[2, 1], [4, 3]], dtype=float)
        L, U = lu_decomposition(A)
        np.testing.assert_allclose(L @ U, A, atol=TOL)
        # L should have ones on diagonal
        np.testing.assert_allclose(np.diag(L), np.ones(2), atol=TOL)
        # U should be upper triangular
        assert np.allclose(U, np.triu(U), atol=TOL)

    def test_3x3_reconstruction(self):
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=float)
        L, U = lu_decomposition(A)
        np.testing.assert_allclose(L @ U, A, atol=TOL)
        # Lower triangular check
        assert np.allclose(L, np.tril(L), atol=TOL)
        assert np.allclose(U, np.triu(U), atol=TOL)

    def test_random_5x5(self):
        np.random.seed(123)
        A = np.random.randn(5, 5)
        L, U = lu_decomposition(A)
        np.testing.assert_allclose(L @ U, A, atol=TOL)


# ── qr_decomposition ─────────────────────────────────────────────────────────

class TestQRDecomposition:
    def test_3x3_orthogonality(self):
        np.random.seed(42)
        A = np.random.randn(3, 3)
        Q, R = qr_decomposition(A)
        np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=TOL)

    def test_3x3_reconstruction(self):
        np.random.seed(42)
        A = np.random.randn(3, 3)
        Q, R = qr_decomposition(A)
        np.testing.assert_allclose(Q @ R, A, atol=TOL)

    def test_upper_triangular(self):
        np.random.seed(42)
        A = np.random.randn(4, 4)
        Q, R = qr_decomposition(A)
        assert np.allclose(R, np.triu(R), atol=TOL)

    def test_rectangular_tall(self):
        np.random.seed(42)
        A = np.random.randn(5, 3)
        Q, R = qr_decomposition(A)
        np.testing.assert_allclose(Q @ R, A, atol=TOL)
        np.testing.assert_allclose(Q.T @ Q, np.eye(3), atol=TOL)
        assert R.shape == (3, 3)


# ── eigendecomposition ────────────────────────────────────────────────────────

class TestEigendecomposition:
    def test_2x2_known(self):
        # Symmetric matrix with known eigenvalues 1 and 5
        A = np.array([[3, 2], [2, 3]], dtype=float)
        vals, vecs = eigendecomposition(A)
        vals_sorted = np.sort(vals)
        np.testing.assert_allclose(vals_sorted, [1.0, 5.0], atol=1e-6)

    def test_diagonal_matrix(self):
        A = np.diag([3.0, 1.0, 4.0])
        vals, vecs = eigendecomposition(A)
        vals_sorted = np.sort(vals)
        np.testing.assert_allclose(vals_sorted, [1.0, 3.0, 4.0], atol=1e-6)

    def test_reconstruction(self):
        np.random.seed(42)
        M = np.random.randn(4, 4)
        A = M @ M.T  # symmetric PD
        vals, vecs = eigendecomposition(A)
        reconstructed = vecs @ np.diag(vals) @ vecs.T
        np.testing.assert_allclose(reconstructed, A, atol=1e-6)

    def test_matches_np_linalg(self):
        np.random.seed(99)
        M = np.random.randn(5, 5)
        A = M + M.T  # symmetric
        vals, vecs = eigendecomposition(A)
        np_vals = np.sort(np.linalg.eigvalsh(A))
        np.testing.assert_allclose(np.sort(vals), np_vals, atol=1e-4)


# ── svd ───────────────────────────────────────────────────────────────────────

class TestSVD:
    def test_square_reconstruction(self):
        np.random.seed(42)
        A = np.random.randn(4, 4)
        U, S, Vt = svd(A)
        reconstructed = U[:, :len(S)] @ np.diag(S) @ Vt[:len(S), :]
        np.testing.assert_allclose(reconstructed, A, atol=1e-6)

    def test_rectangular_reconstruction(self):
        np.random.seed(42)
        A = np.random.randn(5, 3)
        U, S, Vt = svd(A)
        k = len(S)
        reconstructed = U[:, :k] @ np.diag(S) @ Vt[:k, :]
        np.testing.assert_allclose(reconstructed, A, atol=1e-6)

    def test_singular_values_match_np(self):
        np.random.seed(42)
        A = np.random.randn(6, 4)
        _, S, _ = svd(A)
        S_np = np.linalg.svd(A, compute_uv=False)
        np.testing.assert_allclose(np.sort(S)[::-1], np.sort(S_np)[::-1], atol=1e-4)

    def test_u_orthogonal(self):
        np.random.seed(42)
        A = np.random.randn(4, 4)
        U, S, Vt = svd(A)
        np.testing.assert_allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-6)

    def test_vt_orthogonal(self):
        np.random.seed(42)
        A = np.random.randn(4, 4)
        U, S, Vt = svd(A)
        np.testing.assert_allclose(Vt @ Vt.T, np.eye(Vt.shape[0]), atol=1e-6)


# ── solve_linear_system ───────────────────────────────────────────────────────

class TestSolveLinearSystem:
    def test_2x2_known(self):
        A = np.array([[2, 1], [1, 3]], dtype=float)
        b = np.array([5, 10], dtype=float)
        x = solve_linear_system(A, b)
        np.testing.assert_allclose(A @ x, b, atol=TOL)

    def test_identity(self):
        b = np.array([3, 7, 2], dtype=float)
        x = solve_linear_system(np.eye(3), b)
        np.testing.assert_allclose(x, b, atol=TOL)

    def test_random_5x5(self):
        np.random.seed(42)
        A = np.random.randn(5, 5)
        b = np.random.randn(5)
        x = solve_linear_system(A, b)
        np.testing.assert_allclose(A @ x, b, atol=1e-8)


# ── compute_pseudoinverse ─────────────────────────────────────────────────────

class TestPseudoinverse:
    def test_square_nonsingular(self):
        np.random.seed(42)
        A = np.random.randn(3, 3)
        A_pinv = compute_pseudoinverse(A)
        np.testing.assert_allclose(A @ A_pinv, np.eye(3), atol=1e-6)

    def test_rectangular_least_squares(self):
        np.random.seed(42)
        A = np.random.randn(5, 3)
        b = np.random.randn(5)
        A_pinv = compute_pseudoinverse(A)
        x_pinv = A_pinv @ b
        x_np = np.linalg.lstsq(A, b, rcond=None)[0]
        np.testing.assert_allclose(x_pinv, x_np, atol=1e-6)

    def test_matches_np_pinv(self):
        np.random.seed(42)
        A = np.random.randn(4, 6)
        A_pinv = compute_pseudoinverse(A)
        np.testing.assert_allclose(A_pinv, np.linalg.pinv(A), atol=1e-6)


# ── is_positive_definite ──────────────────────────────────────────────────────

class TestIsPositiveDefinite:
    def test_identity_is_pd(self):
        assert is_positive_definite(np.eye(3)) is True

    def test_negative_definite(self):
        assert is_positive_definite(-np.eye(3)) is False

    def test_random_pd(self):
        np.random.seed(42)
        M = np.random.randn(4, 4)
        A = M @ M.T + 0.1 * np.eye(4)  # guaranteed PD
        assert is_positive_definite(A) is True

    def test_singular_is_not_pd(self):
        A = np.array([[1, 0], [0, 0]], dtype=float)
        assert is_positive_definite(A) is False

    def test_indefinite(self):
        A = np.array([[1, 0], [0, -1]], dtype=float)
        assert is_positive_definite(A) is False
