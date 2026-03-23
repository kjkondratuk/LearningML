"""
SVM Exercises -- Spiral 2, Phase 2B, Module 05
"""

import numpy as np
from typing import Callable, Optional


def hard_margin_svm_primal(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, float]:
    """Hard-margin SVM via the primal formulation.

    Minimize 0.5 ||w||^2 subject to y_i (w^T x_i + b) >= 1 for all i.
    Derive: margin = 2/||w||.

    Args:
        X: shape (n, d), data
        y: shape (n,), labels in {-1, +1}

    Returns:
        w: shape (d,), weight vector
        b: bias scalar
    """
    raise NotImplementedError


def hard_margin_svm_dual(
    X: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    """Hard-margin SVM via the dual formulation.

    Derive the dual via Lagrange multipliers:
    max_alpha sum(alpha_i) - 0.5 sum_ij alpha_i alpha_j y_i y_j x_i^T x_j
    s.t. alpha_i >= 0, sum(alpha_i y_i) = 0

    Recover w = sum(alpha_i y_i x_i), b from support vectors.

    Args:
        X: shape (n, d)
        y: shape (n,), labels in {-1, +1}

    Returns:
        w: shape (d,)
        b: scalar
        alphas: shape (n,), Lagrange multipliers
    """
    raise NotImplementedError


def soft_margin_svm(
    X: np.ndarray, y: np.ndarray, C: float
) -> tuple[np.ndarray, float, np.ndarray]:
    """Soft-margin SVM with slack variables.

    Add slack: y_i (w^T x_i + b) >= 1 - xi_i, xi_i >= 0.
    C controls trade-off between margin and violations.

    Args:
        X: shape (n, d)
        y: shape (n,), labels in {-1, +1}
        C: regularization parameter

    Returns:
        w: shape (d,)
        b: scalar
        alphas: shape (n,)
    """
    raise NotImplementedError


def kernel_function(
    x1: np.ndarray,
    x2: np.ndarray,
    kernel_type: str = "rbf",
    **params,
) -> float:
    """Compute kernel k(x1, x2).

    Supported kernels:
    - "linear": x1^T x2
    - "polynomial": (x1^T x2 + c)^degree
    - "rbf": exp(-gamma * ||x1-x2||^2)

    Args:
        x1, x2: input vectors
        kernel_type: "linear", "polynomial", or "rbf"
        **params: gamma (rbf), degree and c (polynomial)

    Returns:
        scalar kernel value
    """
    raise NotImplementedError


def kernel_svm(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    kernel_fn: Callable,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Kernel SVM using the kernel trick in the dual formulation.

    Key insight: replace x_i^T x_j with kernel_fn(x_i, x_j).
    We never compute the high-dimensional feature map explicitly.

    Args:
        X: shape (n, d)
        y: shape (n,), labels in {-1, +1}
        C: regularization
        kernel_fn: kernel function k(x1, x2) -> scalar

    Returns:
        alphas: shape (n,), dual variables
        b: bias
        support_indices: indices of support vectors
    """
    raise NotImplementedError


def smo_simplified(
    X: np.ndarray,
    y: np.ndarray,
    C: float,
    kernel_fn: Callable,
    tol: float = 1e-3,
    max_passes: int = 100,
) -> tuple[np.ndarray, float]:
    """Simplified Sequential Minimal Optimization for SVM training.

    At each step, pick two alpha values and optimize them jointly
    (the only case with a closed-form solution for the dual).

    Args:
        X: shape (n, d)
        y: shape (n,), in {-1, +1}
        C: box constraint
        kernel_fn: kernel function
        tol: KKT tolerance
        max_passes: max passes without alpha changes

    Returns:
        alphas: shape (n,)
        b: bias
    """
    raise NotImplementedError


def predict_svm(
    X_test: np.ndarray,
    X_support: np.ndarray,
    y_support: np.ndarray,
    alphas: np.ndarray,
    b: float,
    kernel_fn: Callable,
) -> np.ndarray:
    """Predict using support vectors only.

    f(x) = sum_sv alpha_i y_i k(x_i, x) + b

    Args:
        X_test: shape (m, d)
        X_support: shape (s, d), support vectors
        y_support: shape (s,), support vector labels
        alphas: shape (s,), non-zero dual variables
        b: bias
        kernel_fn: kernel function

    Returns:
        predictions: shape (m,), in {-1, +1}
    """
    raise NotImplementedError
