"""
Module 02: Neural Net from Scratch
====================================
Build a two-layer neural network with NumPy only.
"""

import numpy as np


def initialize_weights(
    layer_dims: list[int], seed: int = 42
) -> list[dict[str, np.ndarray]]:
    """Initialize weights and biases for a multi-layer network using He initialization.

    For each layer l:
        W_l has shape (layer_dims[l], layer_dims[l-1])
        b_l has shape (layer_dims[l], 1)
        W_l is drawn from N(0, sqrt(2/fan_in))
        b_l is initialized to zeros

    Args:
        layer_dims: List of layer sizes, e.g. [2, 100, 3] for a network
            with 2 inputs, 100 hidden units, and 3 outputs.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts, each with keys 'W' and 'b'.
        Length is len(layer_dims) - 1.
    """
    raise NotImplementedError


def forward_pass(
    X: np.ndarray, params: list[dict[str, np.ndarray]]
) -> tuple[np.ndarray, list[dict]]:
    """Compute the forward pass of a multi-layer ReLU network with softmax output.

    Hidden layers use ReLU activation. The output layer uses softmax.

    Args:
        X: Input data of shape (n_features, n_samples).
        params: List of dicts with 'W' and 'b' for each layer.

    Returns:
        Tuple of:
        - output: Softmax probabilities of shape (n_classes, n_samples).
        - cache: List of dicts with 'Z', 'A' (pre- and post-activation)
          for each layer. cache[0]['A'] should be the input X.
    """
    raise NotImplementedError


def backward_pass(
    Y: np.ndarray,
    output: np.ndarray,
    params: list[dict[str, np.ndarray]],
    cache: list[dict],
) -> list[dict[str, np.ndarray]]:
    """Compute gradients via backpropagation.

    Uses cross-entropy loss with softmax output.

    Args:
        Y: One-hot encoded targets of shape (n_classes, n_samples).
        output: Softmax output from forward_pass, same shape as Y.
        params: Network parameters (for weight matrices in backward step).
        cache: Cached activations from forward_pass.

    Returns:
        List of dicts, each with keys 'dW' and 'db', same structure as params.
    """
    raise NotImplementedError


def update_params(
    params: list[dict[str, np.ndarray]],
    grads: list[dict[str, np.ndarray]],
    lr: float,
) -> list[dict[str, np.ndarray]]:
    """Update parameters using vanilla gradient descent.

    W_new = W_old - lr * dW
    b_new = b_old - lr * db

    Args:
        params: Current parameters.
        grads: Gradients from backward_pass.
        lr: Learning rate.

    Returns:
        Updated parameters (new list, does not modify originals).
    """
    raise NotImplementedError


def train_network(
    X: np.ndarray,
    Y: np.ndarray,
    layer_dims: list[int],
    lr: float = 0.01,
    epochs: int = 1000,
    seed: int = 42,
) -> tuple[list[dict[str, np.ndarray]], list[float]]:
    """Train a multi-layer network end-to-end.

    Args:
        X: Input data of shape (n_features, n_samples).
        Y: One-hot targets of shape (n_classes, n_samples).
        layer_dims: Layer dimensions including input and output.
        lr: Learning rate.
        epochs: Number of training iterations.
        seed: Random seed.

    Returns:
        Tuple of (trained_params, loss_history).
        loss_history is a list of cross-entropy loss values, one per epoch.
    """
    raise NotImplementedError
