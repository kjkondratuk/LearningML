"""
Mini-Project: MNIST Digit Classifier
======================================
Train a feedforward network on MNIST handwritten digits.

Target: >95% test accuracy within 5 epochs.
"""

import torch
import torch.nn as nn


def mnist_digit_classifier(
    hidden_dim: int = 256,
    batch_size: int = 64,
    lr: float = 0.001,
    epochs: int = 5,
) -> dict:
    """Download MNIST, train a feedforward classifier, return results.

    Architecture:
        Flatten(28*28) -> Linear(784, hidden_dim) -> ReLU ->
        Linear(hidden_dim, hidden_dim) -> ReLU ->
        Linear(hidden_dim, 10)

    Uses:
        - torchvision.datasets.MNIST for data
        - CrossEntropyLoss
        - Adam optimizer

    Args:
        hidden_dim: Hidden layer size.
        batch_size: Batch size.
        lr: Learning rate.
        epochs: Number of training epochs.

    Returns:
        Dict with keys:
        - "train_losses": list of average loss per epoch
        - "test_accuracy": float, final test accuracy
        - "model": the trained nn.Module
    """
    raise NotImplementedError
