"""Gaussian Process Exercises. Reference: Rasmussen & Williams ch. 2."""

import torch
import numpy as np


def gp_prior_sample(X: torch.Tensor, kernel_fn: callable, n_samples: int = 5) -> torch.Tensor:
    """Sample functions from GP prior: f ~ N(0, K(X,X)).

    Args:
        X: Input points, shape (n, d).
        kernel_fn: Kernel function.
        n_samples: Number of function samples.

    Returns:
        Function values, shape (n_samples, n).
    """
    raise NotImplementedError


def gp_posterior(X_train, y_train, X_test, kernel_fn, noise_var: float = 1e-4):
    """GP posterior mean and variance.

    Returns:
        Tuple of (mean, variance) at test points.
    """
    raise NotImplementedError


def rbf_kernel_with_hyperparams(x1, x2, length_scale: float = 1.0, signal_var: float = 1.0) -> torch.Tensor:
    """ARD RBF kernel: k(x,y) = signal_var * exp(-||x-y||^2 / (2*l^2))."""
    raise NotImplementedError


def log_marginal_likelihood(X, y, kernel_fn, noise_var: float) -> float:
    """log p(y|X) = -1/2 y^T K_y^{-1} y - 1/2 log|K_y| - n/2 log(2pi)."""
    raise NotImplementedError


def gp_hyperparameter_optimization(X, y, kernel_fn) -> dict:
    """Optimize kernel hyperparams by maximizing marginal likelihood."""
    raise NotImplementedError
