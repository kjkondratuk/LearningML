"""Tests for DDPM mini-project."""

import pytest
import torch

from mini_project import SimpleUNet


class TestSimpleUNet:
    def test_output_shape_matches_input(self):
        model = SimpleUNet(in_channels=1, time_emb_dim=128)
        x = torch.randn(2, 1, 28, 28)
        t = torch.randint(0, 1000, (2,))
        out = model(x, t)
        assert out.shape == x.shape, f"U-Net output shape {out.shape} must match input {x.shape}"

    def test_different_timesteps_different_outputs(self):
        model = SimpleUNet(in_channels=1, time_emb_dim=128)
        x = torch.randn(1, 1, 28, 28)
        out_t10 = model(x, torch.tensor([10]))
        out_t500 = model(x, torch.tensor([500]))
        assert not torch.allclose(out_t10, out_t500, atol=1e-5), (
            "Different timesteps should produce different outputs"
        )
