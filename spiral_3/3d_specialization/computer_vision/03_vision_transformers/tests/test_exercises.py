"""Tests for Vision Transformers."""
import pytest
import torch
from exercises import (
    image_to_patches, patch_embedding, vit_cls_token, vit_forward,
)

class TestPatches:
    def test_correct_number(self):
        img = torch.randn(1, 3, 32, 32)
        patches = image_to_patches(img, patch_size=8)
        assert patches.shape[1] == 16  # (32/8)^2
    def test_reconstruction(self):
        img = torch.randn(1, 3, 32, 32)
        patches = image_to_patches(img, patch_size=8)
        # Should be able to reconstruct original
        assert patches.shape[-1] == 8 * 8 * 3 or patches.numel() == img.numel()
