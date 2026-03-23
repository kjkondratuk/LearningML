"""
Module 01: Perceptron & Neurons
================================
Implement the building blocks of the simplest neural network.
"""

import numpy as np


def step_function(x: float) -> int:
    """Return 1 if x >= 0, else 0.

    Args:
        x: A scalar value.

    Returns:
        1 or 0.

    Example:
        >>> step_function(0.5)
        1
        >>> step_function(-0.1)
        0
    """
    raise NotImplementedError


def perceptron_predict(weights: np.ndarray, bias: float, x: np.ndarray) -> int:
    """Compute the output of a single perceptron.

    output = step_function(weights . x + bias)

    Args:
        weights: 1D array of shape (n_features,).
        bias: Scalar bias term.
        x: 1D array of shape (n_features,).

    Returns:
        0 or 1.
    """
    raise NotImplementedError


def perceptron_train(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    max_epochs: int = 100,
) -> tuple[np.ndarray, float]:
    """Train a single perceptron using the perceptron learning rule.

    For each misclassified sample:
        w = w + lr * (target - predicted) * x
        b = b + lr * (target - predicted)

    Args:
        X: Training data of shape (n_samples, n_features).
        y: Binary targets of shape (n_samples,), values in {0, 1}.
        lr: Learning rate.
        max_epochs: Maximum number of passes over the data.

    Returns:
        Tuple of (weights, bias) after training.
    """
    raise NotImplementedError


def relu(x: np.ndarray) -> np.ndarray:
    """Compute the ReLU activation element-wise.

    relu(x) = max(0, x)

    Args:
        x: Array of any shape.

    Returns:
        Array of the same shape with negative values clipped to 0.
    """
    raise NotImplementedError


def softmax(z: np.ndarray) -> np.ndarray:
    """Compute the softmax of a 1D array of logits.

    softmax(z_i) = exp(z_i) / sum(exp(z_j))

    For numerical stability, subtract max(z) before exponentiating.

    Args:
        z: 1D array of shape (n_classes,).

    Returns:
        1D array of the same shape, values in (0, 1), summing to 1.
    """
    raise NotImplementedError
