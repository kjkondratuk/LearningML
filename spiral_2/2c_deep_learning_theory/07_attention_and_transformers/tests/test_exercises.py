"""
Tests for Attention and Transformer exercises.
"""

import numpy as np
import pytest

from exercises import (
    scaled_dot_product_attention,
    multi_head_attention,
    positional_encoding,
    causal_mask,
    transformer_encoder_layer,
)


class TestScaledDotProductAttention:
    def test_output_shape(self):
        Q = np.random.randn(2, 5, 8)
        K = np.random.randn(2, 7, 8)
        V = np.random.randn(2, 7, 16)
        out, weights = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (2, 5, 16)
        assert weights.shape == (2, 5, 7)

    def test_weights_sum_to_one(self):
        Q = np.random.randn(1, 3, 4)
        K = np.random.randn(1, 5, 4)
        V = np.random.randn(1, 5, 4)
        _, weights = scaled_dot_product_attention(Q, K, V)
        np.testing.assert_allclose(weights.sum(axis=-1), 1.0, atol=1e-6)

    def test_mask_prevents_future(self):
        """Causal mask should zero out attention to future positions."""
        seq_len = 4
        Q = np.random.randn(1, seq_len, 8)
        K = np.random.randn(1, seq_len, 8)
        V = np.random.randn(1, seq_len, 8)
        mask = causal_mask(seq_len)
        _, weights = scaled_dot_product_attention(Q, K, V, mask=mask[None, :, :])
        # Position 0 should only attend to position 0
        np.testing.assert_allclose(weights[0, 0, 1:], 0.0, atol=1e-6)


class TestMultiHeadAttention:
    def test_output_shape(self):
        np.random.seed(42)
        batch, seq, d_model = 2, 10, 16
        n_heads = 4
        Q = np.random.randn(batch, seq, d_model)
        K = np.random.randn(batch, seq, d_model)
        V = np.random.randn(batch, seq, d_model)
        W_q = np.random.randn(d_model, d_model) * 0.1
        W_k = np.random.randn(d_model, d_model) * 0.1
        W_v = np.random.randn(d_model, d_model) * 0.1
        W_o = np.random.randn(d_model, d_model) * 0.1
        out = multi_head_attention(Q, K, V, n_heads, W_q, W_k, W_v, W_o)
        assert out.shape == (batch, seq, d_model)


class TestPositionalEncoding:
    def test_shape(self):
        PE = positional_encoding(100, 64)
        assert PE.shape == (100, 64)

    def test_sin_cos_pattern(self):
        """Even indices should use sin, odd indices should use cos."""
        PE = positional_encoding(10, 4)
        # Position 0: sin(0) = 0 for even dims
        np.testing.assert_allclose(PE[0, 0], 0.0, atol=1e-10)
        # Position 0: cos(0) = 1 for odd dims
        np.testing.assert_allclose(PE[0, 1], 1.0, atol=1e-10)


class TestCausalMask:
    def test_shape(self):
        mask = causal_mask(5)
        assert mask.shape == (5, 5)

    def test_lower_triangular_allowed(self):
        mask = causal_mask(4)
        # Lower triangle (including diagonal) should be 0
        for i in range(4):
            for j in range(i + 1):
                assert mask[i, j] == 0.0

    def test_upper_triangular_masked(self):
        mask = causal_mask(4)
        for i in range(4):
            for j in range(i + 1, 4):
                assert mask[i, j] == float('-inf')
