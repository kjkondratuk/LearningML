"""
Mini-Project: Multi-class Classification on MNIST Subset
=========================================================

Spiral 2, Phase 2B, Module 02

1. Load MNIST digits 0-4 (5 classes)
2. Flatten 28x28 to 784-dim vectors
3. Implement multinomial logistic regression from scratch
4. Train with mini-batch SGD
5. Plot: training loss, confusion matrix, misclassified examples
6. Compare against sklearn's LogisticRegression
"""

import numpy as np


def load_mnist_subset(digits: list[int], n_train: int = 5000, n_test: int = 1000):
    """Load a subset of MNIST digits.

    Args:
        digits: which digits to include (e.g., [0,1,2,3,4])
        n_train: max training samples
        n_test: max test samples

    Returns:
        X_train, y_train, X_test, y_test
    """
    raise NotImplementedError


def train_multinomial_lr(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    lr: float = 0.01,
    batch_size: int = 64,
    n_epochs: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, list[float]]:
    """Train multinomial LR with mini-batch SGD.

    Returns:
        W: shape (d, n_classes), learned weights
        losses: list of loss values per epoch
    """
    raise NotImplementedError


def predict(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Predict class labels.

    Args:
        X: shape (n, d)
        W: shape (d, n_classes)

    Returns:
        predictions: shape (n,), integer class labels
    """
    raise NotImplementedError


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Compute confusion matrix.

    Args:
        y_true: shape (n,)
        y_pred: shape (n,)
        n_classes: number of classes

    Returns:
        cm: shape (n_classes, n_classes), cm[i,j] = count of true=i, pred=j
    """
    raise NotImplementedError


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy."""
    raise NotImplementedError
