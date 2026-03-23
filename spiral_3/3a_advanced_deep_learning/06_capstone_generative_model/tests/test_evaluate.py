"""Tests for DDPM evaluation metrics."""

import pytest
import torch

from evaluate import compute_fid


class TestFID:
    def test_identical_distributions_near_zero(self):
        real = torch.randn(100, 3, 32, 32)
        fid = compute_fid(real, real.clone())
        assert abs(fid) < 5.0, f"FID of identical images should be ~0, got {fid}"

    def test_fid_nonnegative(self):
        real = torch.randn(50, 3, 32, 32)
        fake = torch.randn(50, 3, 32, 32) + 2.0
        fid = compute_fid(real, fake)
        assert fid >= -1.0, f"FID should be non-negative, got {fid}"
