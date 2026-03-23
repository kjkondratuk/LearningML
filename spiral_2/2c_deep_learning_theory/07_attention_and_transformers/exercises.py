"""
Attention and Transformer Exercises -- Spiral 2, Phase 2C, Module 07
"""

import numpy as np


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray,
    mask: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention.

    Derive scaling: if entries of Q,K ~ N(0,1), then Q@K^T has variance d_k.
    Dividing by sqrt(d_k) keeps softmax from saturating.

    attention(Q,K,V) = softmax(Q K^T / sqrt(d_k)) @ V

    Args:
        Q: shape (..., seq_q, d_k)
        K: shape (..., seq_k, d_k)
        V: shape (..., seq_k, d_v)
        mask: shape (..., seq_q, seq_k) or None. -inf where masked.

    Returns:
        output: shape (..., seq_q, d_v)
        attention_weights: shape (..., seq_q, seq_k)
    """
    raise NotImplementedError


def multi_head_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray,
    n_heads: int,
    W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray, W_o: np.ndarray,
    mask: np.ndarray = None,
) -> np.ndarray:
    """Multi-head attention.

    1. Project Q, K, V through W_q, W_k, W_v
    2. Split into n_heads heads
    3. Apply scaled_dot_product_attention to each head
    4. Concatenate heads
    5. Project through W_o

    Derive why multiple heads learn different attention patterns.

    Args:
        Q, K, V: shape (batch, seq, d_model)
        n_heads: number of attention heads
        W_q, W_k, W_v: shape (d_model, d_model)
        W_o: shape (d_model, d_model)
        mask: optional attention mask

    Returns:
        output: shape (batch, seq, d_model)
    """
    raise NotImplementedError


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Derive: PE(pos+k) is a linear function of PE(pos), so the model
    can learn to attend to relative positions.

    Args:
        seq_len: maximum sequence length
        d_model: model dimensionality

    Returns:
        PE: shape (seq_len, d_model)
    """
    raise NotImplementedError


def feed_forward_block(
    X: np.ndarray, W1: np.ndarray, b1: np.ndarray,
    W2: np.ndarray, b2: np.ndarray,
) -> np.ndarray:
    """Position-wise feed-forward network.

    FFN(x) = GELU(x W1 + b1) W2 + b2

    Two linear layers with GELU in between.

    Args:
        X: shape (batch, seq, d_model)
        W1: shape (d_model, d_ff)
        W2: shape (d_ff, d_model)

    Returns:
        output: shape (batch, seq, d_model)
    """
    raise NotImplementedError


def transformer_encoder_layer(
    X: np.ndarray, n_heads: int, d_model: int, d_ff: int, params: dict
) -> np.ndarray:
    """One Transformer encoder layer.

    LayerNorm -> Multi-Head Self-Attention -> Residual ->
    LayerNorm -> FFN -> Residual

    (Pre-norm architecture, as used in modern Transformers.)

    Args:
        X: shape (batch, seq, d_model)
        params: dict with attention and FFN weights, LN parameters

    Returns:
        output: shape (batch, seq, d_model)
    """
    raise NotImplementedError


def transformer_decoder_layer(
    X: np.ndarray, encoder_output: np.ndarray,
    n_heads: int, d_model: int, d_ff: int, params: dict
) -> np.ndarray:
    """One Transformer decoder layer.

    Self-attention (with causal mask) -> cross-attention -> FFN.

    Args:
        X: shape (batch, seq_tgt, d_model)
        encoder_output: shape (batch, seq_src, d_model)
        params: all layer parameters

    Returns:
        output: shape (batch, seq_tgt, d_model)
    """
    raise NotImplementedError


def causal_mask(seq_len: int) -> np.ndarray:
    """Generate upper-triangular causal mask.

    Returns a matrix where position (i, j) is 0 if j <= i (allowed)
    and -inf if j > i (masked / future position).

    Args:
        seq_len: sequence length

    Returns:
        mask: shape (seq_len, seq_len)
    """
    raise NotImplementedError


def full_transformer(
    src: np.ndarray, tgt: np.ndarray,
    n_encoder_layers: int, n_decoder_layers: int,
    params: dict
) -> np.ndarray:
    """Complete encoder-decoder Transformer.

    Args:
        src: source tokens, shape (batch, src_len, d_model)
        tgt: target tokens, shape (batch, tgt_len, d_model)
        params: all model parameters

    Returns:
        output: shape (batch, tgt_len, d_model)
    """
    raise NotImplementedError
