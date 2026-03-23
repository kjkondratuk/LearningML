"""Tests for LM fine-tuning capstone."""

import pytest
import torch
import torch.nn as nn

from finetune import LoRALayer


class TestLoRALayer:
    def test_shape_preserved(self):
        """LoRA layer should produce same output shape as original."""
        original = nn.Linear(256, 256)
        lora = LoRALayer(original, rank=16, alpha=16.0)
        x = torch.randn(4, 256)
        out = lora(x)
        assert out.shape == (4, 256)

    def test_rank_decomposition(self):
        """Low-rank update should have correct dimensions."""
        original = nn.Linear(256, 512)
        lora = LoRALayer(original, rank=8, alpha=8.0)
        # B should be (out_features, rank), A should be (rank, in_features)
        assert hasattr(lora, "lora_A") or hasattr(lora, "A")
        assert hasattr(lora, "lora_B") or hasattr(lora, "B")

    def test_initial_output_matches_original(self):
        """At initialization, LoRA delta should be zero (B initialized to 0)."""
        original = nn.Linear(128, 128)
        lora = LoRALayer(original, rank=4, alpha=4.0)
        x = torch.randn(2, 128)
        original_out = original(x)
        lora_out = lora(x)
        assert torch.allclose(original_out, lora_out, atol=1e-5), (
            "LoRA should not change output at initialization"
        )
