"""
Module 06: RNNs for Sequences
================================
Implement recurrent networks for text classification.
"""

import numpy as np
import torch
import torch.nn as nn


def rnn_cell_manual(
    x: np.ndarray, h_prev: np.ndarray,
    W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray,
) -> np.ndarray:
    """Compute one step of a vanilla RNN cell.

    h_new = tanh(W_xh @ x + W_hh @ h_prev + b_h)

    Args:
        x: Input at current time step, shape (input_size,).
        h_prev: Previous hidden state, shape (hidden_size,).
        W_xh: Input-to-hidden weights, shape (hidden_size, input_size).
        W_hh: Hidden-to-hidden weights, shape (hidden_size, hidden_size).
        b_h: Bias, shape (hidden_size,).

    Returns:
        New hidden state, shape (hidden_size,).
    """
    raise NotImplementedError


def rnn_forward_manual(
    X: np.ndarray, h0: np.ndarray,
    W_xh: np.ndarray, W_hh: np.ndarray, b_h: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Unroll an RNN over a sequence.

    Args:
        X: Input sequence, shape (seq_len, input_size).
        h0: Initial hidden state, shape (hidden_size,).
        W_xh: Input-to-hidden weights, shape (hidden_size, input_size).
        W_hh: Hidden-to-hidden weights, shape (hidden_size, hidden_size).
        b_h: Bias, shape (hidden_size,).

    Returns:
        Tuple of:
        - all_hidden: All hidden states, shape (seq_len, hidden_size).
        - h_final: Final hidden state, shape (hidden_size,).
    """
    raise NotImplementedError


class LSTMClassifier(nn.Module):
    """LSTM-based text classifier.

    Architecture:
        Embedding(vocab_size, embed_dim) ->
        LSTM(embed_dim, hidden_dim, batch_first=True) ->
        Linear(hidden_dim, num_classes)

    Uses the final hidden state for classification.

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Embedding dimension.
        hidden_dim: LSTM hidden state dimension.
        num_classes: Number of output classes.
    """

    def __init__(self, vocab_size: int, embed_dim: int,
                 hidden_dim: int, num_classes: int):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Integer token indices, shape (batch_size, seq_len).

        Returns:
            Logits, shape (batch_size, num_classes).
        """
        raise NotImplementedError


def tokenize_simple(texts: list[str], max_vocab: int = 1000,
                    max_len: int = 50) -> tuple[torch.Tensor, dict[str, int]]:
    """Tokenize a list of texts into padded integer tensors.

    Steps:
        1. Lowercase and split on whitespace.
        2. Build vocabulary from the most common max_vocab words.
           Reserve index 0 for <PAD> and 1 for <UNK>.
        3. Convert each text to a list of token indices.
        4. Pad or truncate to max_len.

    Args:
        texts: List of raw text strings.
        max_vocab: Maximum vocabulary size (including special tokens).
        max_len: Maximum sequence length.

    Returns:
        Tuple of:
        - tokens: LongTensor of shape (len(texts), max_len).
        - vocab: Dict mapping word -> index.
    """
    raise NotImplementedError


def train_sentiment_model(
    texts: list[str],
    labels: list[int],
    epochs: int = 10,
    lr: float = 0.001,
    hidden_dim: int = 64,
) -> dict:
    """Train an LSTM sentiment classifier on the given data.

    Args:
        texts: List of text samples.
        labels: List of integer labels (0 or 1).
        epochs: Number of training epochs.
        lr: Learning rate.
        hidden_dim: LSTM hidden dimension.

    Returns:
        Dict with keys:
        - "model": the trained LSTMClassifier
        - "vocab": the vocabulary dict
        - "train_losses": list of loss per epoch
        - "train_accuracy": final training accuracy
    """
    raise NotImplementedError
