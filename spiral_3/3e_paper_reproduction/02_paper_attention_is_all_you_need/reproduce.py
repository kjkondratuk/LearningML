"""Reproduce: Full Transformer for Machine Translation.

Reference: Vaswani et al. 2017 (arXiv:1706.03762)
"""

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """Transformer encoder with N layers."""
    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048, n_layers: int = 6, dropout: float = 0.1):
        raise NotImplementedError
    def forward(self, src: torch.Tensor, src_mask: torch.Tensor | None = None):
        raise NotImplementedError


class TransformerDecoder(nn.Module):
    """Transformer decoder with N layers and cross-attention."""
    def __init__(self, d_model: int = 512, n_heads: int = 8, d_ff: int = 2048, n_layers: int = 6, dropout: float = 0.1):
        raise NotImplementedError
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        raise NotImplementedError


class Transformer(nn.Module):
    """Full encoder-decoder Transformer for sequence-to-sequence."""
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int = 512, n_heads: int = 8, n_layers: int = 6):
        raise NotImplementedError
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        raise NotImplementedError


def label_smoothing_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 0.1, pad_idx: int = 0) -> torch.Tensor:
    """Cross-entropy with label smoothing."""
    raise NotImplementedError


def transformer_lr_schedule(d_model: int, step: int, warmup_steps: int = 4000) -> float:
    """Learning rate schedule from the paper."""
    raise NotImplementedError
