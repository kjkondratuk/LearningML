"""
Capstone: Image Classifier -- Training Pipeline
=================================================
Train and compare SimpleCNN vs TransferModel on CIFAR-10.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def load_data(batch_size: int = 64, resize_for_transfer: bool = False) -> dict:
    """Download CIFAR-10 and create train/test DataLoaders.

    Applies standard normalization. If resize_for_transfer is True, also
    resizes images to 224x224 for pretrained ResNet compatibility.

    Args:
        batch_size: Batch size for DataLoaders.
        resize_for_transfer: Whether to resize to 224x224.

    Returns:
        Dict with keys:
        - "train_loader": DataLoader for training set
        - "test_loader": DataLoader for test set
        - "classes": tuple of class name strings
    """
    raise NotImplementedError


def train(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int = 10,
    lr: float = 0.001,
) -> dict:
    """Train a model on the given DataLoader.

    Uses CrossEntropyLoss and Adam optimizer.
    Moves model to available device (cuda if available, else cpu).

    Args:
        model: The model to train.
        train_loader: Training DataLoader.
        epochs: Number of epochs.
        lr: Learning rate.

    Returns:
        Dict with keys:
        - "train_losses": list of average loss per epoch
        - "train_accuracies": list of accuracy per epoch
    """
    raise NotImplementedError


def evaluate_and_report(
    model: nn.Module, test_loader: DataLoader
) -> dict:
    """Evaluate the model and compute detailed metrics.

    Args:
        model: Trained model.
        test_loader: Test DataLoader.

    Returns:
        Dict with keys:
        - "test_accuracy": overall accuracy (float)
        - "per_class_accuracy": dict mapping class_name -> accuracy
        - "confusion_matrix": 2D numpy array of shape (num_classes, num_classes)
    """
    raise NotImplementedError


def compare_models(simple_results: dict, transfer_results: dict) -> str:
    """Print and return a comparison table of two model results.

    Args:
        simple_results: Output of evaluate_and_report for SimpleCNN.
        transfer_results: Output of evaluate_and_report for TransferModel.

    Returns:
        A formatted string with the comparison table.
    """
    raise NotImplementedError


if __name__ == "__main__":
    from model import SimpleCNN, TransferModel

    print("=" * 60)
    print("CIFAR-10 Image Classification Capstone")
    print("=" * 60)

    # --- SimpleCNN ---
    print("\n[1/4] Training SimpleCNN from scratch...")
    simple_model = SimpleCNN(num_classes=10)
    data_simple = load_data(batch_size=64, resize_for_transfer=False)
    train(simple_model, data_simple["train_loader"], epochs=15, lr=0.001)
    simple_results = evaluate_and_report(simple_model, data_simple["test_loader"])
    print(f"SimpleCNN test accuracy: {simple_results['test_accuracy']:.1%}")

    # --- TransferModel ---
    print("\n[2/4] Fine-tuning TransferModel (ResNet-18)...")
    transfer_model = TransferModel(num_classes=10, freeze=True)
    data_transfer = load_data(batch_size=32, resize_for_transfer=True)
    train(transfer_model, data_transfer["train_loader"], epochs=5, lr=0.001)
    transfer_results = evaluate_and_report(
        transfer_model, data_transfer["test_loader"]
    )
    print(f"TransferModel test accuracy: {transfer_results['test_accuracy']:.1%}")

    # --- Compare ---
    print("\n[3/4] Comparison:")
    print(compare_models(simple_results, transfer_results))

    print("\n[4/4] Done!")
