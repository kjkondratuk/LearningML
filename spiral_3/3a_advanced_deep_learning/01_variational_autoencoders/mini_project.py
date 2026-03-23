"""
Mini-Project: Full VAE on MNIST

Build a complete Variational Autoencoder and train it on MNIST.

Requirements:
1. Encoder: 784 -> 256 -> (mu, log_var) with latent_dim=20
2. Decoder: 20 -> 256 -> 784
3. Train for 50 epochs, logging ELBO, reconstruction loss, and KL separately
4. Generate samples by decoding z ~ N(0, I)
5. Perform latent space interpolation between two digit classes

Reference: Kingma & Welling 2013 (arXiv:1312.6114)
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """VAE Encoder: x -> (mu, log_var)."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 20):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, log_var) for the approximate posterior q(z|x)."""
        raise NotImplementedError


class Decoder(nn.Module):
    """VAE Decoder: z -> x_recon."""

    def __init__(self, latent_dim: int = 20, hidden_dim: int = 256, output_dim: int = 784):
        raise NotImplementedError

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return reconstructed x (sigmoid outputs for Bernoulli likelihood)."""
        raise NotImplementedError


class VAE(nn.Module):
    """Full Variational Autoencoder."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 256, latent_dim: int = 20):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: encode, reparameterize, decode.

        Returns:
            (x_recon, mu, log_var)
        """
        raise NotImplementedError

    def generate(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate new samples by decoding z ~ N(0, I)."""
        raise NotImplementedError

    def interpolate(
        self, x1: torch.Tensor, x2: torch.Tensor, steps: int = 10
    ) -> torch.Tensor:
        """Interpolate in latent space between two inputs."""
        raise NotImplementedError


def train_vae(model: VAE, train_loader, epochs: int = 50, lr: float = 1e-3):
    """Train the VAE, returning a dict of loss histories."""
    raise NotImplementedError
