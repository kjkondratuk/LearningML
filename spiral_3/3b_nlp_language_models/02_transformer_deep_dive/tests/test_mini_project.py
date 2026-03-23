"""Tests for efficient attention mini-project."""

import pytest
import torch

from mini_project import (
    StandardAttention,
    TiledAttention,
)


class TestAttentionEquivalence:
    def test_tiled_matches_standard(self):
        torch.manual_seed(42)
        x = torch.randn(2, 64, 128)
        std = StandardAttention(d_model=128, num_heads=4)
        tiled = TiledAttention(d_model=128, num_heads=4, block_size=16)
        # Copy weights
        tiled.load_state_dict(std.state_dict())
        out_std = std(x)
        out_tiled = tiled(x)
        assert torch.allclose(out_std, out_tiled, atol=1e-4)
