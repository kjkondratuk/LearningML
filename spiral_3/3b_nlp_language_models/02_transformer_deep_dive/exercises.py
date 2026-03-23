"""
Transformer Deep Dive Exercises

Implement attention and positional encoding components from:
    Vaswani et al. (2017), arXiv:1706.03762
    Su et al. (2021), arXiv:2104.09864
    Dao et al. (2022), arXiv:2205.14135
"""

import torch


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """Sinusoidal positional encoding from Vaswani et al. 2017.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        seq_len: Maximum sequence length.
        d_model: Model dimension.

    Returns:
        Positional encoding matrix, shape (seq_len, d_model).
    """
    raise NotImplementedError


def rotary_position_embedding(x: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Apply Rotary Position Embedding (RoPE) to query or key tensor.

    RoPE encodes position by rotating pairs of dimensions using position-dependent
    rotation matrices. The dot product between RoPE-encoded q and k depends only
    on relative position.

    Reference: Su et al. 2021 (arXiv:2104.09864)

    Args:
        x: Query or key tensor, shape (batch_size, seq_len, d_model).
        seq_len: Sequence length.

    Returns:
        Rotated tensor, same shape as x.
    """
    raise NotImplementedError


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: float = 0.0,
) -> torch.Tensor:
    """Scaled dot-product attention.

    Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V

    Args:
        q: Queries, shape (..., seq_len_q, d_k).
        k: Keys, shape (..., seq_len_k, d_k).
        v: Values, shape (..., seq_len_k, d_v).
        mask: Optional mask, shape (..., seq_len_q, seq_len_k). True = masked (ignore).
        dropout: Dropout probability on attention weights.

    Returns:
        Attention output, shape (..., seq_len_q, d_v).
    """
    raise NotImplementedError


def multi_head_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    d_model: int,
    W_q: torch.Tensor | None = None,
    W_k: torch.Tensor | None = None,
    W_v: torch.Tensor | None = None,
    W_o: torch.Tensor | None = None,
) -> torch.Tensor:
    """Multi-head attention: split into heads, attend, concatenate, project.

    Args:
        q, k, v: Input tensors, shape (batch_size, seq_len, d_model).
        num_heads: Number of attention heads.
        d_model: Model dimension.
        W_q, W_k, W_v: Projection matrices, shape (d_model, d_model). Optional.
        W_o: Output projection, shape (d_model, d_model). Optional.

    Returns:
        Output tensor, shape (batch_size, seq_len_q, d_model).
    """
    raise NotImplementedError


def flash_attention_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_size: int = 64,
) -> torch.Tensor:
    """Simplified tiled attention implementing FlashAttention's core idea.

    Process attention in tiles to reduce memory from O(n^2) to O(n).
    Uses the online softmax trick to compute exact attention tile by tile.

    Reference: Dao et al. 2022, FlashAttention (arXiv:2205.14135)

    Args:
        q: Queries, shape (batch_size, seq_len, d_k).
        k: Keys, shape (batch_size, seq_len, d_k).
        v: Values, shape (batch_size, seq_len, d_v).
        block_size: Tile size for blocked computation.

    Returns:
        Attention output, shape (batch_size, seq_len, d_v).
    """
    raise NotImplementedError


def alibi_bias(num_heads: int, seq_len: int) -> torch.Tensor:
    """Compute ALiBi positional bias matrix.

    Each head gets a different slope m. bias_ij = -m * |i - j|.
    Slopes are geometric: m_h = 1 / 2^(8h/H) for head h of H total.

    Reference: Press et al. 2022, "Train Short, Test Long" (arXiv:2108.12409)

    Args:
        num_heads: Number of attention heads.
        seq_len: Sequence length.

    Returns:
        Bias tensor, shape (num_heads, seq_len, seq_len).
    """
    raise NotImplementedError
