"""
Capstone: Image Classifier -- Model Definitions
=================================================
Two models for CIFAR-10: a CNN from scratch and a transfer learning model.
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """A 3-block CNN for CIFAR-10 (32x32 RGB images).

    Architecture:
        Conv2d(3, 32, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Conv2d(32, 64, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Conv2d(64, 128, 3, padding=1) -> ReLU -> MaxPool2d(2)
        Flatten -> Linear(128*4*4, 256) -> ReLU -> Dropout(0.5) -> Linear(256, 10)

    Target: >70% test accuracy on CIFAR-10.
    """

    def __init__(self, num_classes: int = 10):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        raise NotImplementedError


class TransferModel(nn.Module):
    """A pretrained ResNet-18 adapted for CIFAR-10.

    Loads ResNet-18 with ImageNet weights, freezes the backbone,
    and replaces the final FC layer for num_classes outputs.

    Target: >85% test accuracy on CIFAR-10.
    """

    def __init__(self, num_classes: int = 10, freeze: bool = True):
        """Initialize the transfer model.

        Args:
            num_classes: Number of output classes.
            freeze: If True, freeze all backbone parameters.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, H, W).
                For best results, H=W=224 (ImageNet size).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        raise NotImplementedError
