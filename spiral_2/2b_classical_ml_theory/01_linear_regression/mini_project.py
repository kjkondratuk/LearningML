"""
Mini-Project: Bayesian Linear Regression with Uncertainty
==========================================================

Spiral 2, Phase 2B, Module 01

1. Generate synthetic data from a known linear function + noise
2. Fit Bayesian linear regression
3. Plot: data, posterior mean, 1-sigma and 2-sigma uncertainty bands
4. Sequential updating: 5, 10, 20, 100 points -- show posterior sharpening
5. Compare Bayesian uncertainty to bootstrap uncertainty
"""

import numpy as np


def generate_synthetic_data(
    n: int, w_true: np.ndarray, noise_std: float, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate y = Xw + epsilon, epsilon ~ N(0, noise_std^2).

    Args:
        n: number of data points
        w_true: true weight vector, shape (d,)
        noise_std: standard deviation of noise
        seed: random seed

    Returns:
        X: shape (n, d), design matrix
        y: shape (n,), targets
    """
    raise NotImplementedError


def sequential_bayesian_update(
    X: np.ndarray,
    y: np.ndarray,
    batch_sizes: list[int],
    prior_precision: float,
    noise_precision: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fit Bayesian regression with increasing amounts of data.

    For each batch_size in batch_sizes, use the first batch_size points
    and return the posterior (mean, covariance).

    Args:
        X: full design matrix
        y: full target vector
        batch_sizes: list of how many points to use, e.g. [5, 10, 20, 100]
        prior_precision: prior precision
        noise_precision: noise precision

    Returns:
        list of (mu_post, Sigma_post) tuples, one per batch size
    """
    raise NotImplementedError


def bootstrap_uncertainty(
    X: np.ndarray,
    y: np.ndarray,
    X_test: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate prediction uncertainty via bootstrap.

    Args:
        X: training design matrix
        y: training targets
        X_test: test points
        n_bootstrap: number of bootstrap samples
        seed: random seed

    Returns:
        pred_mean: shape (m,), mean of bootstrap predictions
        pred_std: shape (m,), std of bootstrap predictions
    """
    raise NotImplementedError
