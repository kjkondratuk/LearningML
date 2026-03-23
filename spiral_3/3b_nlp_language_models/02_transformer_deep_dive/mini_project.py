"""
Mini-Project: Efficient Attention Variants

Compare standard attention, FlashAttention-style tiled attention,
and linear attention on speed and memory.

References:
    Dao et al. 2022, FlashAttention (arXiv:2205.14135)
    Katharopoulos et al. 2020, Linear Transformers (arXiv:2006.16236)
"""

import torch
import torch.nn as nn


class StandardAttention(nn.Module):
    """Naive O(n^2) attention for baseline comparison."""

    def __init__(self, d_model: int, num_heads: int):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class TiledAttention(nn.Module):
    """Memory-efficient tiled attention (FlashAttention concept)."""

    def __init__(self, d_model: int, num_heads: int, block_size: int = 64):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def benchmark_attention(seq_lengths: list[int], d_model: int = 256, num_heads: int = 8):
    """Benchmark memory and speed of attention variants."""
    raise NotImplementedError
