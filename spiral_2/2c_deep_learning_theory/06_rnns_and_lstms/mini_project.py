"""
Mini-Project: Character-Level Language Model
==============================================

Spiral 2, Phase 2C, Module 06

Train a character-level LSTM language model on text:
1. Implement the LSTM model
2. Train on Shakespeare or similar corpus
3. Generate text by sampling
4. Compare RNN vs LSTM generated text quality
"""

import numpy as np


def prepare_char_data(text: str) -> tuple[dict, dict, np.ndarray]:
    """Convert text to integer sequences with char-to-int mapping.

    Returns:
        char_to_idx: dict
        idx_to_char: dict
        encoded: np.ndarray of int
    """
    raise NotImplementedError


def train_char_lstm(
    data: np.ndarray,
    vocab_size: int,
    hidden_size: int = 128,
    seq_length: int = 100,
    n_epochs: int = 50,
    lr: float = 0.001,
    seed: int = 42,
) -> tuple[dict, list[float]]:
    """Train LSTM language model.

    Returns:
        params: trained parameters
        losses: training loss per epoch
    """
    raise NotImplementedError


def generate_text(
    params: dict,
    idx_to_char: dict,
    char_to_idx: dict,
    seed_text: str,
    length: int = 200,
    temperature: float = 1.0,
) -> str:
    """Generate text by sampling from the model.

    Args:
        temperature: controls randomness (lower = more deterministic)

    Returns:
        generated string
    """
    raise NotImplementedError
