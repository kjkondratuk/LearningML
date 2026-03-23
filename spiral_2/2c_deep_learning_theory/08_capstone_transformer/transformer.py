"""
transformer.py -- Complete Transformer Implementation
=======================================================

Spiral 2, Phase 2C, Module 08 (Capstone)

Build from scratch in PyTorch. No nn.Transformer or nn.TransformerEncoder.
Supports encoder-only (BERT) and decoder-only (GPT) configurations.
"""

import numpy as np


class MultiHeadAttention:
    """Multi-head scaled dot-product attention.

    Implements: attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
    with n_heads parallel attention heads.
    """

    def __init__(self, d_model: int, n_heads: int):
        raise NotImplementedError

    def forward(self, Q, K, V, mask=None):
        raise NotImplementedError


class PositionalEncoding:
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class FeedForward:
    """Position-wise feed-forward: Linear -> GELU -> Linear."""

    def __init__(self, d_model: int, d_ff: int):
        raise NotImplementedError

    def forward(self, x):
        raise NotImplementedError


class TransformerBlock:
    """One Transformer block: LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, causal: bool = False):
        raise NotImplementedError

    def forward(self, x, encoder_output=None, mask=None):
        raise NotImplementedError


class TransformerLM:
    """Decoder-only Transformer language model (GPT-style).

    Embedding -> Positional Encoding -> N x TransformerBlock -> LayerNorm -> Linear
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 512,
    ):
        raise NotImplementedError

    def forward(self, x):
        """Forward pass. Returns logits over vocabulary.

        Args:
            x: token indices, shape (batch, seq_len)

        Returns:
            logits: shape (batch, seq_len, vocab_size)
        """
        raise NotImplementedError

    def generate(self, prompt, max_tokens: int = 100, temperature: float = 1.0):
        """Autoregressive text generation."""
        raise NotImplementedError


class TransformerEncoder:
    """Encoder-only Transformer (BERT-style).

    Embedding -> Positional Encoding -> N x TransformerBlock -> LayerNorm
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        max_len: int = 512,
    ):
        raise NotImplementedError

    def forward(self, x):
        """Returns hidden states for all positions."""
        raise NotImplementedError
