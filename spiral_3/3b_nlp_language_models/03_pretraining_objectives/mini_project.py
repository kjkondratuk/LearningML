"""
Mini-Project: Pretrain a Small Transformer on WikiText

Compare MLM vs CLM pretraining on the same architecture.
"""

import torch
import torch.nn as nn


class SmallTransformer(nn.Module):
    """Small transformer for pretraining experiments."""

    def __init__(self, vocab_size: int, d_model: int = 256, n_heads: int = 4, n_layers: int = 4):
        raise NotImplementedError

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
        raise NotImplementedError


def pretrain_mlm(model, train_loader, epochs: int = 10, mask_prob: float = 0.15):
    """Pretrain with masked language modeling."""
    raise NotImplementedError


def pretrain_clm(model, train_loader, epochs: int = 10):
    """Pretrain with causal language modeling."""
    raise NotImplementedError
