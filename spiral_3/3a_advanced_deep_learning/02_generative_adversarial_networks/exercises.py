"""
Generative Adversarial Network Exercises

Implement GAN components from:
    Goodfellow et al. (2014) arXiv:1406.2661
    Arjovsky et al. (2017) arXiv:1701.07875
    Gulrajani et al. (2017) arXiv:1704.00028
"""

import torch
import torch.nn as nn


def generator_loss_original(d_fake: torch.Tensor) -> torch.Tensor:
    """Original GAN generator loss: -E[log(D(G(z)))].

    This is the non-saturating variant that provides better gradients early in training.

    Args:
        d_fake: Discriminator outputs on generated samples, shape (batch_size, 1).

    Returns:
        Scalar loss.
    """
    raise NotImplementedError


def discriminator_loss_original(
    d_real: torch.Tensor, d_fake: torch.Tensor
) -> torch.Tensor:
    """Original GAN discriminator loss.

    loss = -E[log(D(x))] - E[log(1 - D(G(z)))]

    Args:
        d_real: Discriminator outputs on real samples, shape (batch_size, 1).
        d_fake: Discriminator outputs on fake samples, shape (batch_size, 1).

    Returns:
        Scalar loss.
    """
    raise NotImplementedError


def wasserstein_loss(d_real: torch.Tensor, d_fake: torch.Tensor) -> torch.Tensor:
    """Wasserstein distance estimate (critic loss for WGAN).

    loss = E[D(G(z))] - E[D(x)]

    Note: the critic has no sigmoid activation. Minimizing this pushes
    D(real) up and D(fake) down, estimating the Wasserstein distance.

    Args:
        d_real: Critic outputs on real samples, shape (batch_size, 1).
        d_fake: Critic outputs on fake samples, shape (batch_size, 1).

    Returns:
        Scalar loss (to be minimized by the critic).
    """
    raise NotImplementedError


def gradient_penalty(
    discriminator: nn.Module,
    real: torch.Tensor,
    fake: torch.Tensor,
    lambda_gp: float = 10.0,
) -> torch.Tensor:
    """WGAN-GP gradient penalty to enforce 1-Lipschitz constraint.

    Interpolate between real and fake: x_hat = alpha * real + (1 - alpha) * fake
    Penalize ||grad_x_hat D(x_hat)||_2 deviating from 1.

    penalty = lambda * E[(||grad D(x_hat)||_2 - 1)^2]

    Reference: Gulrajani et al. 2017 (arXiv:1704.00028)

    Args:
        discriminator: The critic network.
        real: Real samples, shape (batch_size, *).
        fake: Generated samples, shape (batch_size, *).
        lambda_gp: Gradient penalty coefficient (default 10).

    Returns:
        Scalar gradient penalty term.
    """
    raise NotImplementedError


def spectral_norm(weight_matrix: torch.Tensor) -> torch.Tensor:
    """Compute the spectral norm of a weight matrix.

    The spectral norm is the largest singular value: sigma_1(W).
    Used for spectral normalization (Miyato et al. 2018, arXiv:1802.05957).

    Args:
        weight_matrix: 2D weight tensor, shape (out_features, in_features).

    Returns:
        Scalar spectral norm.
    """
    raise NotImplementedError


def frechet_inception_distance(
    real_features: torch.Tensor, fake_features: torch.Tensor
) -> float:
    """Compute the Frechet Inception Distance (FID) between two distributions.

    FID = ||mu_r - mu_f||^2 + Tr(Sigma_r + Sigma_f - 2*(Sigma_r @ Sigma_f)^{1/2})

    Assumes features are extracted from a pretrained network (e.g., Inception v3).

    Reference: Heusel et al. 2017, "GANs Trained by a Two Time-Scale Update Rule"
               (arXiv:1706.08500)

    Args:
        real_features: Features from real images, shape (n_real, feature_dim).
        fake_features: Features from generated images, shape (n_fake, feature_dim).

    Returns:
        FID score (float). Lower is better. 0 means identical distributions.
    """
    raise NotImplementedError
