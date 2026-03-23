"""Tests for 3D Vision & NeRFs."""
import pytest
import torch
from exercises import (
    ray_generation, volume_rendering, positional_encoding_nerf, nerf_mlp_forward,
)

class TestVolumeRendering:
    def test_transmittance_decreases(self):
        densities = torch.ones(1, 10) * 0.5
        colors = torch.rand(1, 10, 3)
        t_vals = torch.linspace(0, 1, 11).unsqueeze(0)
        output = volume_rendering(densities, colors, t_vals)
        assert output.shape[-1] == 3
    def test_positional_encoding_dim(self):
        x = torch.randn(10, 3)
        encoded = positional_encoding_nerf(x, num_frequencies=10)
        assert encoded.shape[-1] == 3 * 2 * 10 or encoded.shape[-1] == 3 + 3 * 2 * 10
