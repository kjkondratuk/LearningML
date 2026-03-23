"""
Tests for Self-Supervised Learning exercises.

Verifies:
- InfoNCE loss decreases with better positive alignment
- NT-Xent temperature effects
- MAE loss is 0 for perfect reconstruction
- BYOL loss symmetry
"""

import pytest
import torch

from exercises import (
    byol_loss,
    info_nce_loss,
    masked_autoencoder_loss,
    nt_xent_loss,
)


class TestInfoNCELoss:
    def test_loss_decreases_with_alignment(self):
        """Loss should decrease when positive pair similarity increases."""
        anchor = torch.randn(16, 64)
        anchor = anchor / anchor.norm(dim=1, keepdim=True)
        negatives = torch.randn(16, 32, 64)

        # Bad positive: random
        pos_bad = torch.randn(16, 64)
        loss_bad = info_nce_loss(anchor, pos_bad, negatives, temperature=0.1)

        # Good positive: close to anchor
        pos_good = anchor + 0.01 * torch.randn_like(anchor)
        loss_good = info_nce_loss(anchor, pos_good, negatives, temperature=0.1)

        assert loss_good < loss_bad, (
            "InfoNCE loss should decrease with better positive alignment"
        )

    def test_loss_nonnegative(self):
        anchor = torch.randn(8, 32)
        positive = torch.randn(8, 32)
        negatives = torch.randn(8, 16, 32)
        loss = info_nce_loss(anchor, positive, negatives)
        assert loss.item() >= -1e-6, "InfoNCE loss should be non-negative"

    def test_lower_temperature_sharper(self):
        """Lower temperature should produce more concentrated distributions."""
        anchor = torch.randn(16, 64)
        positive = anchor + 0.1 * torch.randn_like(anchor)
        negatives = torch.randn(16, 32, 64)

        loss_low_t = info_nce_loss(anchor, positive, negatives, temperature=0.1)
        loss_high_t = info_nce_loss(anchor, positive, negatives, temperature=1.0)
        # With a good positive, lower temperature makes the loss smaller
        assert loss_low_t < loss_high_t


class TestNTXentLoss:
    def test_identical_pairs_low_loss(self):
        """Identical views should give very low loss."""
        z = torch.randn(16, 64)
        z = z / z.norm(dim=1, keepdim=True)
        loss = nt_xent_loss(z, z.clone(), temperature=0.1)
        # Not exactly 0 because other pairs in batch serve as pseudo-negatives
        assert loss.item() < 2.0, f"Loss for identical pairs should be low, got {loss.item()}"

    def test_temperature_effect(self):
        """Lower temperature should give sharper softmax distributions."""
        z_i = torch.randn(32, 64)
        z_j = z_i + 0.1 * torch.randn_like(z_i)

        loss_sharp = nt_xent_loss(z_i, z_j, temperature=0.1)
        loss_smooth = nt_xent_loss(z_i, z_j, temperature=1.0)
        assert loss_sharp < loss_smooth, (
            "Lower temperature should give lower loss when pairs are aligned"
        )


class TestMAELoss:
    def test_perfect_reconstruction(self):
        """Loss should be 0 when reconstruction matches original on masked regions."""
        original = torch.randn(4, 16, 48)
        mask = torch.randint(0, 2, (4, 16)).float()
        loss = masked_autoencoder_loss(original, original.clone(), mask)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6), (
            "Loss should be 0 for perfect reconstruction"
        )

    def test_only_masked_positions_matter(self):
        """Unmasked positions should not affect the loss."""
        original = torch.randn(4, 16, 48)
        reconstructed = original.clone()
        mask = torch.zeros(4, 16)
        mask[:, :8] = 1  # First 8 patches are masked

        # Corrupt unmasked positions -- should not affect loss
        reconstructed[:, 8:] = torch.randn_like(reconstructed[:, 8:])
        loss = masked_autoencoder_loss(original, reconstructed, mask)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    def test_loss_nonnegative(self):
        original = torch.randn(4, 16, 48)
        reconstructed = torch.randn(4, 16, 48)
        mask = torch.ones(4, 16)
        loss = masked_autoencoder_loss(original, reconstructed, mask)
        assert loss.item() >= 0


class TestBYOLLoss:
    def test_identical_projections(self):
        """Loss should be 0 for identical projections."""
        proj = torch.randn(8, 64)
        loss = byol_loss(proj, proj.clone())
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)

    def test_orthogonal_projections_higher_loss(self):
        """Orthogonal vectors should give higher loss than aligned ones."""
        online = torch.randn(8, 64)
        online = online / online.norm(dim=1, keepdim=True)
        target_aligned = online + 0.01 * torch.randn_like(online)
        target_orthogonal = torch.randn(8, 64)

        loss_aligned = byol_loss(online, target_aligned)
        loss_orthogonal = byol_loss(online, target_orthogonal)
        assert loss_aligned < loss_orthogonal

    def test_loss_nonnegative(self):
        online = torch.randn(8, 64)
        target = torch.randn(8, 64)
        loss = byol_loss(online, target)
        assert loss.item() >= -1e-6
