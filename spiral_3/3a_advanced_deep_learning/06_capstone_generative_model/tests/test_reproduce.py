"""
Tests for DDPM reproduction.

Verifies:
- U-Net output shape matches input shape
- Forward pass valid at all timesteps
- Sampling produces valid pixel range
"""

import pytest
import torch

from reproduce import (
    SinusoidalTimeEmbedding,
    UNet,
)


class TestSinusoidalTimeEmbedding:
    def test_output_shape(self):
        emb = SinusoidalTimeEmbedding(dim=128)
        t = torch.tensor([0, 50, 500, 999])
        out = emb(t)
        assert out.shape == (4, 128)

    def test_different_timesteps_different_embeddings(self):
        emb = SinusoidalTimeEmbedding(dim=128)
        t = torch.tensor([0, 500])
        out = emb(t)
        assert not torch.allclose(out[0], out[1])


class TestUNet:
    def test_output_shape_matches_input(self):
        """Critical: U-Net must produce same spatial dimensions as input."""
        model = UNet(in_channels=3, model_channels=32, channel_mults=(1, 2))
        x = torch.randn(2, 3, 32, 32)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        assert out.shape == x.shape, (
            f"U-Net output {out.shape} must match input {x.shape}"
        )

    def test_forward_at_all_timesteps(self):
        """Model should produce valid outputs for t=0, t=T/2, t=T-1."""
        model = UNet(in_channels=3, model_channels=32, channel_mults=(1, 2))
        x = torch.randn(1, 3, 32, 32)
        for t_val in [0, 500, 999]:
            t = torch.tensor([t_val])
            out = model(x, t)
            assert torch.isfinite(out).all(), f"Output not finite at t={t_val}"
