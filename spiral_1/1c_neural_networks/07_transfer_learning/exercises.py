"""
Module 07: Transfer Learning
==============================
Leverage pretrained models for new tasks.
"""

import torch
import torch.nn as nn


def load_pretrained_resnet() -> nn.Module:
    """Load a pretrained ResNet-18 from torchvision.

    Uses weights="IMAGENET1K_V1".

    Returns:
        A ResNet-18 model with pretrained ImageNet weights.
    """
    raise NotImplementedError


def freeze_backbone(model: nn.Module, num_classes: int = 10) -> nn.Module:
    """Freeze all parameters and replace the final FC layer.

    Steps:
        1. Set requires_grad=False for all parameters.
        2. Replace model.fc with nn.Linear(model.fc.in_features, num_classes).

    The new FC layer should have requires_grad=True (default).

    Args:
        model: A ResNet model.
        num_classes: Number of output classes.

    Returns:
        The modified model (same object, mutated in place).
    """
    raise NotImplementedError


def count_trainable_params(model: nn.Module) -> int:
    """Count the number of parameters with requires_grad=True.

    Args:
        model: Any nn.Module.

    Returns:
        Number of trainable parameters.
    """
    raise NotImplementedError


def unfreeze_last_n_layers(model: nn.Module, n: int) -> nn.Module:
    """Unfreeze the last n children of the model.

    Use list(model.children()) to enumerate children, then set
    requires_grad=True for all parameters in the last n children.

    Args:
        model: A model with frozen backbone.
        n: Number of children to unfreeze from the end.

    Returns:
        The modified model (mutated in place).
    """
    raise NotImplementedError


def fine_tune(
    model: nn.Module,
    num_classes: int = 10,
    epochs: int = 5,
    batch_size: int = 32,
    lr: float = 0.001,
) -> dict:
    """Fine-tune the model on CIFAR-10.

    Steps:
        1. Load CIFAR-10 with appropriate transforms (resize to 224x224,
           normalize with ImageNet stats).
        2. Use CrossEntropyLoss and Adam optimizer.
        3. Train for `epochs` epochs.
        4. Evaluate on test set.

    Args:
        model: A ResNet model (possibly with frozen backbone).
        num_classes: Number of CIFAR-10 classes.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.

    Returns:
        Dict with keys:
        - "train_losses": list of average loss per epoch
        - "test_accuracy": float
    """
    raise NotImplementedError
