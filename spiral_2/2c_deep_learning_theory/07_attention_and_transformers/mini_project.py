"""
Mini-Project: Transformer from Scratch (in PyTorch)
=====================================================

Spiral 2, Phase 2C, Module 07

Build a Transformer using from-scratch components (not nn.Transformer).
Train on sequence sorting: input [3,1,4,1,5] -> output [1,1,3,4,5].
Visualize attention patterns.
"""

import numpy as np


def build_sort_dataset(
    n_samples: int, seq_len: int, vocab_size: int, seed: int = 42
) -> tuple:
    """Generate sorting task data.

    Input: random sequences of integers
    Output: same integers, sorted

    Returns:
        src: shape (n_samples, seq_len), input sequences
        tgt: shape (n_samples, seq_len), sorted sequences
    """
    raise NotImplementedError


def train_sorting_transformer(
    src_train, tgt_train, src_val, tgt_val,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    n_epochs: int = 50,
    lr: float = 1e-3,
) -> tuple:
    """Train Transformer on the sorting task.

    Returns:
        model: trained model
        losses: training losses
        val_accs: validation accuracies
    """
    raise NotImplementedError


def visualize_attention(model, src_sample) -> dict:
    """Extract attention weights for visualization.

    Returns:
        dict mapping layer_name -> attention_weights array
    """
    raise NotImplementedError
