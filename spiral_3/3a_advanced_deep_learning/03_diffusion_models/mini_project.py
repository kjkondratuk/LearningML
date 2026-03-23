"""
Mini-Project: DDPM on MNIST from Scratch

Implement the full DDPM pipeline: U-Net, training, and sampling.

Reference: Ho et al. 2020 (arXiv:2006.11239)
"""

import torch
import torch.nn as nn


class SimpleUNet(nn.Module):
    """Simplified U-Net with time embeddings for DDPM on MNIST."""

    def __init__(self, in_channels: int = 1, time_emb_dim: int = 128):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise given noisy image and timestep.

        Args:
            x: Noisy image, shape (B, C, H, W).
            t: Timestep, shape (B,).

        Returns:
            Predicted noise, same shape as x.
        """
        raise NotImplementedError


def train_ddpm(model: SimpleUNet, train_loader, T: int = 1000, epochs: int = 50):
    """Train DDPM, returning loss history."""
    raise NotImplementedError


def sample_ddpm(model: SimpleUNet, n_samples: int, T: int = 1000, img_shape=(1, 28, 28)):
    """Generate samples using the full DDPM reverse process."""
    raise NotImplementedError
