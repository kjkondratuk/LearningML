"""
Mini-Project: CNN from Scratch in NumPy
=========================================

Spiral 2, Phase 2C, Module 05

Build: Conv(3x3, 8) -> ReLU -> MaxPool(2x2) -> Conv(3x3, 16) -> ReLU -> MaxPool(2x2) -> FC -> Softmax
Train on MNIST, achieve >95% test accuracy.
Visualize learned filters and feature maps.
"""

import numpy as np


def build_mnist_cnn(seed: int = 42) -> dict:
    """Build the 3-layer CNN architecture for MNIST."""
    raise NotImplementedError


def train_cnn(
    model: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.01,
) -> tuple[dict, list[float], list[float]]:
    """Train the CNN.

    Returns:
        model: trained model
        train_losses: per-epoch training losses
        val_accs: per-epoch validation accuracies
    """
    raise NotImplementedError


def visualize_filters(model: dict, layer: str = "conv1") -> np.ndarray:
    """Extract learned filters for visualization.

    Returns: array of filter images
    """
    raise NotImplementedError


def visualize_feature_maps(model: dict, X_sample: np.ndarray) -> list[np.ndarray]:
    """Get feature maps at each layer for a sample image.

    Returns: list of feature map arrays
    """
    raise NotImplementedError
