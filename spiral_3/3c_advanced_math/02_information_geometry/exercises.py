"""
Information Geometry Exercises

Implement Fisher information and natural gradient from:
    Amari (2016), "Information Geometry and Its Applications"
    Martens (2014), arXiv:1412.1193
"""

import torch
import numpy as np


def fisher_information_gaussian(mu: float, sigma: float) -> np.ndarray:
    """Fisher information matrix for N(mu, sigma^2).

    F = [[1/sigma^2, 0], [0, 2/sigma^2]]

    Args:
        mu: Mean parameter.
        sigma: Standard deviation parameter.

    Returns:
        Fisher information matrix, shape (2, 2).
    """
    raise NotImplementedError


def fisher_information_bernoulli(p: float) -> float:
    """Fisher information for Bernoulli(p).

    I(p) = 1 / (p * (1-p))

    Args:
        p: Success probability, in (0, 1).

    Returns:
        Scalar Fisher information.
    """
    raise NotImplementedError


def natural_gradient_step(
    params: torch.Tensor,
    grad: torch.Tensor,
    fisher_matrix: torch.Tensor,
    lr: float,
) -> torch.Tensor:
    """Natural gradient update: params -= lr * F^{-1} @ grad.

    Args:
        params: Current parameters, shape (d,).
        grad: Gradient of loss, shape (d,).
        fisher_matrix: Fisher information matrix, shape (d, d).
        lr: Learning rate.

    Returns:
        Updated parameters, shape (d,).
    """
    raise NotImplementedError


def kl_divergence_second_order(
    fisher: torch.Tensor,
    delta_theta: torch.Tensor,
) -> torch.Tensor:
    """Second-order approximation of KL divergence.

    KL(theta || theta + delta) ~ (1/2) * delta^T @ F @ delta

    Args:
        fisher: Fisher information matrix, shape (d, d).
        delta_theta: Parameter perturbation, shape (d,).

    Returns:
        Approximate KL divergence (scalar).
    """
    raise NotImplementedError


def geodesic_distance_gaussians(
    mu1: float, sigma1: float, mu2: float, sigma2: float
) -> float:
    """Fisher-Rao geodesic distance between two univariate Gaussians.

    d(N(mu1, s1^2), N(mu2, s2^2)) = sqrt(2) * sqrt(
        log((s1^2 + s2^2 + (mu1-mu2)^2) / (2*s1*s2)) - ... (exact formula)
    )

    For the special case of same variance (sigma1 = sigma2 = sigma):
    d = |mu1 - mu2| / sigma

    Args:
        mu1, sigma1: Parameters of first Gaussian.
        mu2, sigma2: Parameters of second Gaussian.

    Returns:
        Fisher-Rao distance (float).
    """
    raise NotImplementedError
