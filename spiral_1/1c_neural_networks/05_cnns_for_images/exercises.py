"""
Module 05: CNNs for Images
============================
Understand convolutions and build a CNN from scratch in PyTorch.
"""

import numpy as np
import torch
import torch.nn as nn


def conv2d_manual(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Apply a 2D convolution (no padding, stride=1) using explicit loops.

    Args:
        image: 2D array of shape (H, W).
        kernel: 2D array of shape (kH, kW).

    Returns:
        2D array of shape (H - kH + 1, W - kW + 1).
    """
    raise NotImplementedError


def compute_output_size(input_size: int, kernel_size: int,
                        padding: int = 0, stride: int = 1) -> int:
    """Compute the output spatial dimension after a conv or pool operation.

    Formula: (input_size - kernel_size + 2 * padding) // stride + 1

    Args:
        input_size: Spatial dimension of input (height or width).
        kernel_size: Size of the kernel.
        padding: Zero-padding added to both sides.
        stride: Stride of the operation.

    Returns:
        Output spatial dimension (int).
    """
    raise NotImplementedError


def build_cnn(num_classes: int = 10) -> nn.Module:
    """Build a CNN for 32x32 RGB images (e.g. CIFAR-10).

    Architecture:
        Conv2d(3, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Conv2d(64, 128, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Flatten -> Linear(128*4*4, 256) -> ReLU -> Linear(256, num_classes)

    Args:
        num_classes: Number of output classes.

    Returns:
        An nn.Module.
    """
    raise NotImplementedError


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: Any nn.Module.

    Returns:
        Dict with keys "total" and "trainable".
    """
    raise NotImplementedError


def train_cnn_on_cifar(
    model: nn.Module,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 0.001,
) -> dict:
    """Train the CNN on CIFAR-10 and return results.

    Uses:
        - torchvision.datasets.CIFAR10 with standard normalization
        - CrossEntropyLoss
        - Adam optimizer

    Args:
        model: The CNN model.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.

    Returns:
        Dict with keys:
        - "train_losses": list of average loss per epoch
        - "test_accuracy": float, final test set accuracy
    """
    raise NotImplementedError
