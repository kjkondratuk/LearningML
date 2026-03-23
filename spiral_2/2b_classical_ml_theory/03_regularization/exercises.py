"""
Regularization Exercises -- Spiral 2, Phase 2B, Module 03
"""

import numpy as np
from typing import Callable


def ridge_regression(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge regression (L2 regularization).

    Derive as MAP with Gaussian prior.
    w = (X^T X + alpha * I)^{-1} X^T y

    Args:
        X: shape (n, d)
        y: shape (n,)
        alpha: regularization strength (prior_precision)

    Returns:
        w: shape (d,)
    """
    raise NotImplementedError


def lasso_coordinate_descent(
    X: np.ndarray, y: np.ndarray, alpha: float, n_steps: int
) -> np.ndarray:
    """Lasso regression (L1) via coordinate descent.

    For each feature j, holding all others fixed, the update is the
    soft-thresholding operator:
        w_j = S(rho_j, alpha) / z_j
    where rho_j = X_j^T (y - X_{-j} w_{-j}) and z_j = X_j^T X_j.

    Args:
        X: shape (n, d)
        y: shape (n,)
        alpha: L1 penalty strength
        n_steps: number of full coordinate passes

    Returns:
        w: shape (d,)
    """
    raise NotImplementedError


def elastic_net(
    X: np.ndarray, y: np.ndarray, alpha: float, l1_ratio: float, n_steps: int
) -> np.ndarray:
    """Elastic net: L1 + L2 regularization.

    Loss = ||y - Xw||^2 + alpha * (l1_ratio * ||w||_1 + (1-l1_ratio) * 0.5 * ||w||_2^2)

    Args:
        X: shape (n, d)
        y: shape (n,)
        alpha: overall regularization strength
        l1_ratio: 0 = pure ridge, 1 = pure lasso
        n_steps: coordinate descent iterations

    Returns:
        w: shape (d,)
    """
    raise NotImplementedError


def regularization_path(
    X: np.ndarray, y: np.ndarray, alphas: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ridge and lasso solutions for a range of alphas.

    Args:
        X: shape (n, d)
        y: shape (n,)
        alphas: shape (K,), regularization strengths

    Returns:
        ridge_weights: shape (K, d)
        lasso_weights: shape (K, d)
    """
    raise NotImplementedError


def l1_sparsity_demo(
    X: np.ndarray, y: np.ndarray, alpha: float
) -> tuple[np.ndarray, int]:
    """Show L1 recovers true features from noisy data.

    Args:
        X: shape (n, d) with d >> true features
        y: shape (n,)
        alpha: L1 penalty

    Returns:
        w: shape (d,), lasso weights
        n_nonzero: number of non-zero weights
    """
    raise NotImplementedError


def cross_validate_alpha(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    k_folds: int,
    model_fn: Callable,
) -> tuple[float, np.ndarray]:
    """K-fold cross-validation to select regularization strength.

    Args:
        X: shape (n, d)
        y: shape (n,)
        alphas: candidate alpha values
        k_folds: number of folds
        model_fn: callable(X_train, y_train, alpha) -> w

    Returns:
        best_alpha: the alpha with lowest CV error
        cv_errors: shape (len(alphas),), mean CV error for each alpha
    """
    raise NotImplementedError
