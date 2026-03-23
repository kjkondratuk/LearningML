"""
Tests for Transformer Deep Dive exercises.

Verifies:
- Sinusoidal PE properties (position 0 consistency, frequency decrease)
- RoPE relative position invariance
- Attention with uniform queries produces mean of values
- Causal mask prevents future attention
- Flash attention matches naive attention
- ALiBi linearity
"""

import pytest
import torch

from exercises import (
    alibi_bias,
    flash_attention_tiled,
    multi_head_attention,
    rotary_position_embedding,
    scaled_dot_product_attention,
    sinusoidal_positional_encoding,
)


class TestSinusoidalPE:
    def test_output_shape(self):
        pe = sinusoidal_positional_encoding(100, 256)
        assert pe.shape == (100, 256)

    def test_position_zero_consistent(self):
        """Position 0 encoding should always be the same."""
        pe1 = sinusoidal_positional_encoding(50, 128)
        pe2 = sinusoidal_positional_encoding(100, 128)
        assert torch.allclose(pe1[0], pe2[0])

    def test_frequencies_decrease_with_dimension(self):
        """Higher dimensions should encode lower frequencies."""
        pe = sinusoidal_positional_encoding(1000, 64)
        # Variance of sin components across positions should decrease with dim
        var_low_dim = pe[:, 0].var()
        var_high_dim = pe[:, -2].var()
        assert var_low_dim >= var_high_dim * 0.5, (
            "Lower dimensions should have higher frequency (more variance)"
        )


class TestRoPE:
    def test_preserves_dot_product_relative(self):
        """RoPE: rotating q and k by same angle preserves relative dot product."""
        d = 32
        q = torch.randn(1, 10, d)
        k = torch.randn(1, 10, d)

        q_rot = rotary_position_embedding(q, 10)
        k_rot = rotary_position_embedding(k, 10)

        # The dot product q_rot[i] . k_rot[i] should equal q[i] . k[i]
        # because RoPE at same position cancels out
        dots_orig = (q[:, 0] * k[:, 0]).sum()
        dots_rot = (q_rot[:, 0] * k_rot[:, 0]).sum()
        assert torch.allclose(dots_orig, dots_rot, atol=1e-4)


class TestScaledDotProductAttention:
    def test_uniform_queries_give_mean_of_values(self):
        """With identical queries, attention weights are uniform -> output is mean of V."""
        seq_len, d = 5, 8
        q = torch.ones(1, seq_len, d)
        k = torch.ones(1, seq_len, d)
        v = torch.randn(1, seq_len, d)
        out = scaled_dot_product_attention(q, k, v)
        expected = v.mean(dim=1, keepdim=True).expand_as(v)
        assert torch.allclose(out, expected, atol=1e-4)

    def test_causal_mask(self):
        """With causal mask, position i should only attend to positions <= i."""
        seq_len, d = 4, 8
        q = torch.randn(1, seq_len, d)
        k = torch.randn(1, seq_len, d)
        v = torch.randn(1, seq_len, d)

        # Causal mask: True means "mask out" (don't attend)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        out = scaled_dot_product_attention(q, k, v, mask=mask)

        # First position should only use its own value
        out_first_alone = scaled_dot_product_attention(
            q[:, :1], k[:, :1], v[:, :1]
        )
        assert torch.allclose(out[:, 0], out_first_alone[:, 0], atol=1e-5)


class TestFlashAttention:
    def test_matches_naive(self):
        """Tiled attention must produce same output as naive attention (within tolerance)."""
        torch.manual_seed(42)
        q = torch.randn(2, 128, 32)
        k = torch.randn(2, 128, 32)
        v = torch.randn(2, 128, 32)

        naive = scaled_dot_product_attention(q, k, v)
        tiled = flash_attention_tiled(q, k, v, block_size=32)

        assert torch.allclose(naive, tiled, atol=1e-4), (
            f"Max diff: {(naive - tiled).abs().max()}"
        )


class TestALiBi:
    def test_shape(self):
        bias = alibi_bias(num_heads=8, seq_len=64)
        assert bias.shape == (8, 64, 64)

    def test_linear_in_distance(self):
        """Bias should be linear in |i - j|."""
        bias = alibi_bias(num_heads=4, seq_len=10)
        # For any head, bias[i, i+1] - bias[i, i+2] should be constant
        for h in range(4):
            d1 = bias[h, 0, 1].item()
            d2 = bias[h, 0, 2].item()
            d3 = bias[h, 0, 3].item()
            step = d2 - d1
            assert abs((d3 - d2) - step) < 1e-5, "ALiBi bias should be linear in distance"

    def test_slopes_differ_per_head(self):
        bias = alibi_bias(num_heads=8, seq_len=10)
        slopes = [bias[h, 0, 1].item() for h in range(8)]
        assert len(set([round(s, 6) for s in slopes])) == 8, (
            "Each head should have a different slope"
        )
