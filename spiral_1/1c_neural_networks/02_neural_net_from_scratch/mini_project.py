"""
Mini-Project: Classify Spirals
===============================
Generate a 2D spiral dataset and train a two-layer network to classify it.

Target: >90% accuracy with a hidden layer of 100 units.
"""

import numpy as np


def generate_spirals(
    n_points: int = 200, n_classes: int = 2, noise: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a 2D spiral dataset.

    Args:
        n_points: Number of points per class.
        n_classes: Number of spiral arms (classes).
        noise: Standard deviation of Gaussian noise added to points.
        seed: Random seed.

    Returns:
        Tuple of:
        - X: shape (2, n_points * n_classes) -- 2D coordinates.
        - Y: shape (n_classes, n_points * n_classes) -- one-hot labels.
    """
    raise NotImplementedError


def classify_spirals(
    hidden_size: int = 100, lr: float = 0.5, epochs: int = 5000, seed: int = 42
) -> tuple[float, list[float]]:
    """Generate spirals, train a network, and return accuracy + loss history.

    Uses a [2, hidden_size, 2] architecture.

    Args:
        hidden_size: Number of hidden units.
        lr: Learning rate.
        epochs: Number of training epochs.
        seed: Random seed.

    Returns:
        Tuple of (accuracy, loss_history).
        accuracy is a float in [0, 1].
    """
    raise NotImplementedError
