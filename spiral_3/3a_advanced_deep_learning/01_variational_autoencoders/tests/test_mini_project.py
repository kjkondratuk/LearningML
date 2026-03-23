"""
Tests for VAE mini-project.

Verifies architectural properties and generation capability.
"""

import pytest
import torch

from mini_project import (
    VAE,
    Decoder,
    Encoder,
)


class TestEncoder:
    def test_output_shapes(self):
        enc = Encoder(input_dim=784, hidden_dim=256, latent_dim=20)
        x = torch.randn(8, 784)
        mu, log_var = enc(x)
        assert mu.shape == (8, 20)
        assert log_var.shape == (8, 20)


class TestDecoder:
    def test_output_shape(self):
        dec = Decoder(latent_dim=20, hidden_dim=256, output_dim=784)
        z = torch.randn(8, 20)
        x_recon = dec(z)
        assert x_recon.shape == (8, 784)

    def test_output_range(self):
        """Decoder outputs should be in [0, 1] (sigmoid)."""
        dec = Decoder(latent_dim=20, hidden_dim=256, output_dim=784)
        z = torch.randn(8, 20)
        x_recon = dec(z)
        assert (x_recon >= 0).all() and (x_recon <= 1).all()


class TestVAE:
    def test_forward_shapes(self):
        model = VAE(input_dim=784, hidden_dim=256, latent_dim=20)
        x = torch.randn(8, 784)
        x_recon, mu, log_var = model(x)
        assert x_recon.shape == (8, 784)
        assert mu.shape == (8, 20)
        assert log_var.shape == (8, 20)

    def test_generate_shape(self):
        model = VAE(input_dim=784, hidden_dim=256, latent_dim=20)
        samples = model.generate(16, device=torch.device("cpu"))
        assert samples.shape == (16, 784)

    def test_interpolate_shape(self):
        model = VAE(input_dim=784, hidden_dim=256, latent_dim=20)
        x1 = torch.randn(1, 784)
        x2 = torch.randn(1, 784)
        interp = model.interpolate(x1, x2, steps=10)
        assert interp.shape[0] == 10
        assert interp.shape[1] == 784
