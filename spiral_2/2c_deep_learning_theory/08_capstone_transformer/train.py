"""
train.py -- Training script for the Transformer
=================================================

Spiral 2, Phase 2C, Module 08 (Capstone)

Usage:
    python train.py --data <path> --epochs 50 --d_model 128 --n_heads 4
"""

import numpy as np


def load_and_tokenize(path: str, max_len: int = 512) -> tuple:
    """Load text data and tokenize (character-level).

    Returns:
        tokens: array of token indices
        vocab: dict mapping char -> index
        inv_vocab: dict mapping index -> char
    """
    raise NotImplementedError


def create_batches(
    tokens: np.ndarray,
    batch_size: int,
    seq_len: int,
    seed: int = 42,
) -> list:
    """Create training batches (input, target) for language modeling.

    Target is input shifted by one position.
    """
    raise NotImplementedError


def train_epoch(model, batches, optimizer, loss_fn) -> float:
    """Train for one epoch. Returns average loss."""
    raise NotImplementedError


def evaluate(model, batches, loss_fn) -> float:
    """Evaluate on validation set. Returns average loss."""
    raise NotImplementedError


def train(
    data_path: str,
    n_epochs: int = 50,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
    d_ff: int = 512,
    batch_size: int = 32,
    seq_len: int = 128,
    lr: float = 3e-4,
    seed: int = 42,
) -> dict:
    """Full training pipeline.

    Returns:
        dict with 'model', 'train_losses', 'val_losses', 'vocab', 'inv_vocab'
    """
    raise NotImplementedError
