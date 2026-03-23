"""
Module 01: Linear Algebra Essentials
======================================
The matrix operations behind every ML algorithm.
"""

import numpy as np


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Compute the dot product of two 1D arrays.

    Implement WITHOUT using np.dot -- use a loop or sum of element-wise product.

    Args:
        a: 1D array of shape (n,).
        b: 1D array of shape (n,).

    Returns:
        Scalar dot product.
    """
    raise NotImplementedError


def matrix_multiply(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Multiply two matrices using explicit loops (the slow way).

    Implement with three nested for-loops. This is the "wrong way" --
    compare performance with np.matmul on large matrices.

    Args:
        A: 2D array of shape (m, n).
        B: 2D array of shape (n, p).

    Returns:
        2D array of shape (m, p).
    """
    raise NotImplementedError


def linear_transform_2d(
    A: np.ndarray, points: np.ndarray
) -> np.ndarray:
    """Apply a 2D linear transformation to a set of points.

    Each column of `points` is a 2D point. Apply the transform: y = A @ x.

    Args:
        A: Transformation matrix of shape (2, 2).
        points: 2D array of shape (2, n_points).

    Returns:
        Transformed points of shape (2, n_points).
    """
    raise NotImplementedError


def compute_covariance_matrix(X: np.ndarray) -> np.ndarray:
    """Compute the covariance matrix of a data matrix.

    Steps:
        1. Center the data (subtract column means).
        2. Compute C = (1 / (n-1)) * X_centered.T @ X_centered.

    Args:
        X: Data matrix of shape (n_samples, n_features).

    Returns:
        Covariance matrix of shape (n_features, n_features).
    """
    raise NotImplementedError


def eigen_decomposition(
    A: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of a symmetric matrix.

    Use np.linalg.eigh (optimized for symmetric matrices).
    Return eigenvalues sorted in DESCENDING order, with eigenvectors
    reordered to match.

    Args:
        A: Symmetric 2D array of shape (n, n).

    Returns:
        Tuple of:
        - eigenvalues: 1D array of shape (n,), descending order.
        - eigenvectors: 2D array of shape (n, n), columns are eigenvectors.
    """
    raise NotImplementedError


def project_onto_principal_components(
    X: np.ndarray, n_components: int
) -> tuple[np.ndarray, float]:
    """Perform PCA: project data onto the top-n principal components.

    Steps:
        1. Center the data.
        2. Compute covariance matrix.
        3. Eigendecompose.
        4. Project onto top n_components eigenvectors.
        5. Compute explained variance ratio.

    Args:
        X: Data matrix of shape (n_samples, n_features).
        n_components: Number of principal components to keep.

    Returns:
        Tuple of:
        - projected: shape (n_samples, n_components).
        - explained_variance_ratio: fraction of total variance explained
          (float in [0, 1]).
    """
    raise NotImplementedError
