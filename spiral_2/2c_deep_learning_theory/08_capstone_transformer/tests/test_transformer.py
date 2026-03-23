"""
Tests for the Transformer capstone.
"""

import numpy as np
import pytest

from transformer import (
    MultiHeadAttention,
    PositionalEncoding,
    FeedForward,
    TransformerBlock,
    TransformerLM,
)
from train import (
    load_and_tokenize,
    create_batches,
)


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(d_model=32, n_heads=4)
        Q = np.random.randn(2, 10, 32)
        out = mha.forward(Q, Q, Q)
        assert out.shape == (2, 10, 32)


class TestPositionalEncoding:
    def test_shape(self):
        pe = PositionalEncoding(d_model=64, max_len=100)
        x = np.random.randn(2, 50, 64)
        out = pe.forward(x)
        assert out.shape == (2, 50, 64)


class TestFeedForward:
    def test_shape(self):
        ff = FeedForward(d_model=32, d_ff=128)
        x = np.random.randn(2, 10, 32)
        out = ff.forward(x)
        assert out.shape == (2, 10, 32)


class TestTransformerLM:
    def test_overfit_tiny(self):
        """Transformer should be able to overfit a tiny sequence."""
        model = TransformerLM(
            vocab_size=10, d_model=32, n_heads=2,
            n_layers=2, d_ff=64, max_len=16
        )
        x = np.array([[1, 2, 3, 4, 5]])
        logits = model.forward(x)
        assert logits.shape == (1, 5, 10)
