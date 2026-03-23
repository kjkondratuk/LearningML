"""
Module 04: The Training Loop
==============================
Master the canonical PyTorch training pipeline.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def create_dataset(
    n_samples: int = 500, n_features: int = 10, n_classes: int = 3, seed: int = 42
) -> Dataset:
    """Create a simple synthetic classification dataset.

    Generate random features from N(0, 1) and assign labels based on the
    sign of the sum of features (or any deterministic rule that creates
    n_classes balanced-ish classes).

    Return a torch Dataset where __getitem__ returns (features_tensor, label_tensor).

    Args:
        n_samples: Number of samples.
        n_features: Number of features per sample.
        n_classes: Number of classes.
        seed: Random seed.

    Returns:
        A torch.utils.data.Dataset instance.
    """
    raise NotImplementedError


def create_dataloader(dataset: Dataset, batch_size: int = 32,
                      shuffle: bool = True) -> DataLoader:
    """Wrap a Dataset in a DataLoader.

    Args:
        dataset: A torch Dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle.

    Returns:
        A DataLoader.
    """
    raise NotImplementedError


def build_classifier(input_dim: int, hidden_dim: int,
                     output_dim: int) -> nn.Module:
    """Build a feedforward classifier.

    Architecture:
        Linear(input_dim, hidden_dim) -> ReLU ->
        Linear(hidden_dim, hidden_dim) -> ReLU ->
        Linear(hidden_dim, output_dim)

    Args:
        input_dim: Number of input features.
        hidden_dim: Number of hidden units per layer.
        output_dim: Number of output classes.

    Returns:
        An nn.Module (use nn.Sequential).
    """
    raise NotImplementedError


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    """Train the model for one epoch and return average loss.

    Steps per batch:
        1. optimizer.zero_grad()
        2. Forward pass
        3. Compute loss
        4. loss.backward()
        5. optimizer.step()

    Args:
        model: The classifier.
        dataloader: Training DataLoader.
        criterion: Loss function (e.g. CrossEntropyLoss).
        optimizer: Optimizer (e.g. Adam).

    Returns:
        Average loss over all batches (float).
    """
    raise NotImplementedError


def evaluate(model: nn.Module, dataloader: DataLoader) -> float:
    """Evaluate the model and return accuracy.

    Use torch.no_grad() and model.eval().

    Args:
        model: The classifier.
        dataloader: Evaluation DataLoader.

    Returns:
        Accuracy as a float in [0, 1].
    """
    raise NotImplementedError


def full_training_loop(
    input_dim: int = 10,
    hidden_dim: int = 64,
    output_dim: int = 3,
    n_samples: int = 1000,
    batch_size: int = 32,
    lr: float = 0.001,
    epochs: int = 20,
    seed: int = 42,
) -> dict:
    """Run the complete training pipeline.

    Steps:
        1. Create dataset and split 80/20 into train/test.
        2. Create DataLoaders.
        3. Build classifier, loss, optimizer.
        4. Train for `epochs` epochs.
        5. Evaluate on test set.

    Args:
        input_dim: Feature dimension.
        hidden_dim: Hidden layer size.
        output_dim: Number of classes.
        n_samples: Total samples.
        batch_size: Batch size.
        lr: Learning rate.
        epochs: Number of epochs.
        seed: Random seed.

    Returns:
        Dict with keys:
        - "train_losses": list of average loss per epoch
        - "test_accuracy": float, final test accuracy
        - "model": the trained nn.Module
    """
    raise NotImplementedError
