"""
Mini-Project: WGAN-GP on CIFAR-10

Implement a Wasserstein GAN with Gradient Penalty and train on CIFAR-10.

References:
    Arjovsky et al. 2017 (arXiv:1701.07875)
    Gulrajani et al. 2017 (arXiv:1704.00028)
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """DCGAN-style generator: z -> image."""

    def __init__(self, latent_dim: int = 128, img_channels: int = 3):
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from noise vectors."""
        raise NotImplementedError


class Critic(nn.Module):
    """WGAN critic (no sigmoid): image -> scalar score."""

    def __init__(self, img_channels: int = 3):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Score an image (higher = more real)."""
        raise NotImplementedError


def train_wgan_gp(
    generator: Generator,
    critic: Critic,
    train_loader,
    epochs: int = 100,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
) -> dict:
    """Train WGAN-GP, returning loss histories and FID scores."""
    raise NotImplementedError
