"""
Tests for GAN exercises.

Verifies:
- Adversarial loss properties
- Wasserstein loss is unbounded
- Gradient penalty enforces 1-Lipschitz
- Spectral norm equals largest singular value
- FID properties
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from exercises import (
    discriminator_loss_original,
    frechet_inception_distance,
    generator_loss_original,
    gradient_penalty,
    spectral_norm,
    wasserstein_loss,
)


class TestOriginalGANLoss:
    def test_generator_loss_decreases_with_better_fakes(self):
        """Generator loss should decrease as discriminator is more fooled."""
        d_fake_bad = torch.sigmoid(torch.tensor([[-2.0, -1.0, -3.0]]).T)
        d_fake_good = torch.sigmoid(torch.tensor([[2.0, 1.0, 3.0]]).T)
        loss_bad = generator_loss_original(d_fake_bad)
        loss_good = generator_loss_original(d_fake_good)
        assert loss_good < loss_bad, (
            "Generator loss should be lower when D(G(z)) is higher"
        )

    def test_discriminator_losses_are_adversarial(self):
        """Improving one player should worsen the other."""
        d_real = torch.sigmoid(torch.tensor([[3.0, 2.0]]).T)
        d_fake_low = torch.sigmoid(torch.tensor([[-2.0, -3.0]]).T)
        d_fake_high = torch.sigmoid(torch.tensor([[2.0, 3.0]]).T)

        d_loss_good = discriminator_loss_original(d_real, d_fake_low)
        d_loss_bad = discriminator_loss_original(d_real, d_fake_high)
        assert d_loss_good < d_loss_bad, (
            "Discriminator loss should be lower when fakes score low"
        )


class TestWassersteinLoss:
    def test_wasserstein_loss_sign(self):
        """Critic loss is negative when D separates real from fake well."""
        d_real = torch.tensor([[5.0, 4.0, 6.0]]).T
        d_fake = torch.tensor([[-5.0, -4.0, -6.0]]).T
        loss = wasserstein_loss(d_real, d_fake)
        assert loss < 0, "Wasserstein critic loss should be negative when well-separated"

    def test_wasserstein_loss_is_unbounded(self):
        """Unlike original GAN loss, Wasserstein loss can take arbitrarily large values."""
        d_real = torch.tensor([[100.0]]).T
        d_fake = torch.tensor([[-100.0]]).T
        loss = wasserstein_loss(d_real, d_fake)
        assert abs(loss.item()) > 10, "Wasserstein loss should be unbounded"


class TestGradientPenalty:
    def test_penalty_near_zero_for_1lipschitz(self):
        """A linear function with unit-norm weights has gradient norm ~1."""
        # Simple linear critic: D(x) = w^T x, ||w|| = 1
        model = nn.Linear(10, 1, bias=False)
        with torch.no_grad():
            model.weight.div_(model.weight.norm())

        real = torch.randn(16, 10, requires_grad=True)
        fake = torch.randn(16, 10, requires_grad=True)
        gp = gradient_penalty(model, real.detach(), fake.detach(), lambda_gp=1.0)
        # For a linear model with unit-norm weights, grad norm is exactly 1
        assert gp.item() < 1.0, (
            f"GP should be small for 1-Lipschitz function, got {gp.item()}"
        )

    def test_penalty_is_nonnegative(self):
        model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
        real = torch.randn(8, 10)
        fake = torch.randn(8, 10)
        gp = gradient_penalty(model, real, fake, lambda_gp=10.0)
        assert gp.item() >= 0, "Gradient penalty must be non-negative"


class TestSpectralNorm:
    def test_matches_svd(self):
        """Spectral norm should match the largest singular value from SVD."""
        torch.manual_seed(42)
        W = torch.randn(10, 5)
        sn = spectral_norm(W)
        _, s, _ = torch.linalg.svd(W)
        expected = s[0]
        assert torch.allclose(sn, expected, atol=1e-4), (
            f"Spectral norm {sn.item()} should match SVD {expected.item()}"
        )

    def test_spectral_norm_positive(self):
        W = torch.randn(8, 8)
        sn = spectral_norm(W)
        assert sn > 0, "Spectral norm must be positive for non-zero matrices"

    def test_identity_spectral_norm_is_one(self):
        W = torch.eye(5)
        sn = spectral_norm(W)
        assert torch.allclose(sn, torch.tensor(1.0), atol=1e-5)


class TestFID:
    def test_identical_distributions(self):
        """FID between identical feature distributions should be 0."""
        features = torch.randn(500, 64)
        fid = frechet_inception_distance(features, features.clone())
        assert abs(fid) < 1.0, f"FID of identical distributions should be ~0, got {fid}"

    def test_fid_nonnegative(self):
        real = torch.randn(200, 64)
        fake = torch.randn(200, 64) + 5.0
        fid = frechet_inception_distance(real, fake)
        assert fid >= -1e-6, f"FID must be non-negative, got {fid}"

    def test_fid_increases_with_shift(self):
        real = torch.randn(300, 32)
        fake_close = torch.randn(300, 32) + 0.1
        fake_far = torch.randn(300, 32) + 5.0
        fid_close = frechet_inception_distance(real, fake_close)
        fid_far = frechet_inception_distance(real, fake_far)
        assert fid_far > fid_close, "FID should increase with distributional shift"
