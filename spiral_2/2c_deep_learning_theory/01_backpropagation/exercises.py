"""
Backpropagation Exercises -- Spiral 2, Phase 2C, Module 01
"""

import numpy as np
from typing import Callable


def forward_pass(
    X: np.ndarray,
    weights: list[np.ndarray],
    biases: list[np.ndarray],
    activations: list[Callable],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Generic forward pass through L layers.

    For each layer l:
        z_l = a_{l-1} @ W_l + b_l
        a_l = activation_l(z_l)

    Store all intermediate z and a values (needed for backprop).

    Args:
        X: shape (batch, d_in), input data
        weights: list of L weight matrices, W_l has shape (d_{l-1}, d_l)
        biases: list of L bias vectors
        activations: list of L activation functions

    Returns:
        output: final layer output
        cached_z: list of pre-activation values at each layer
        cached_a: list of post-activation values (including input X as a_0)
    """
    raise NotImplementedError


def backward_pass(
    y_true: np.ndarray,
    weights: list[np.ndarray],
    cached_z: list[np.ndarray],
    cached_a: list[np.ndarray],
    activation_grads: list[Callable],
    loss_grad_fn: Callable,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Backpropagation: compute gradients for all weights and biases.

    Derive using the chain rule:
    delta_L = loss_grad * activation_grad_L(z_L)
    delta_l = (delta_{l+1} @ W_{l+1}^T) * activation_grad_l(z_l)
    dL/dW_l = a_{l-1}^T @ delta_l
    dL/db_l = sum(delta_l, axis=0)

    Args:
        y_true: shape (batch, d_out), true labels
        weights: list of weight matrices
        cached_z: pre-activation values from forward pass
        cached_a: post-activation values from forward pass
        activation_grads: list of activation gradient functions
        loss_grad_fn: gradient of loss w.r.t. final output

    Returns:
        weight_grads: list of dL/dW for each layer
        bias_grads: list of dL/db for each layer
    """
    raise NotImplementedError


def numerical_gradient_check(
    f: Callable,
    params: list[np.ndarray],
    epsilon: float = 1e-5,
) -> list[np.ndarray]:
    """Verify analytical gradients using numerical differences.

    For each parameter element, compute:
        (f(param + eps) - f(param - eps)) / (2 * eps)

    Compare against analytical gradients. Relative error should be < 1e-5.

    Args:
        f: function mapping params -> scalar loss
        params: list of parameter arrays
        epsilon: perturbation size

    Returns:
        numerical_grads: list of gradient arrays (same shapes as params)
    """
    raise NotImplementedError


def mse_loss_and_grad(
    y_pred: np.ndarray, y_true: np.ndarray
) -> tuple[float, np.ndarray]:
    """MSE loss and its gradient w.r.t. y_pred.

    L = (1/n) sum (y_pred - y_true)^2
    dL/dy_pred = (2/n) (y_pred - y_true)

    Args:
        y_pred, y_true: shape (batch, d)

    Returns:
        (loss, grad)
    """
    raise NotImplementedError


def cross_entropy_loss_and_grad(
    y_pred: np.ndarray, y_true: np.ndarray
) -> tuple[float, np.ndarray]:
    """Cross-entropy loss and gradient.

    L = -(1/n) sum y_true * log(y_pred)
    Assumes y_pred is already softmax output, y_true is one-hot.

    Args:
        y_pred: shape (batch, n_classes), probabilities
        y_true: shape (batch, n_classes), one-hot

    Returns:
        (loss, grad)
    """
    raise NotImplementedError


def softmax_cross_entropy_backward(
    logits: np.ndarray, y_true: np.ndarray
) -> np.ndarray:
    """Combined softmax + cross-entropy gradient.

    Derive: dL/d_logits = softmax(logits) - y_one_hot.
    This elegant simplification avoids computing the Jacobian of softmax.

    Args:
        logits: shape (batch, n_classes), raw scores
        y_true: shape (batch, n_classes), one-hot labels

    Returns:
        grad: shape (batch, n_classes)
    """
    raise NotImplementedError


def train_neural_network(
    X: np.ndarray,
    y: np.ndarray,
    layer_sizes: list[int],
    lr: float,
    n_epochs: int,
    batch_size: int = 32,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Train a multi-layer network from scratch.

    Put it all together: forward, loss, backward, update.

    Args:
        X: shape (n, d_in)
        y: shape (n,) or (n, d_out)
        layer_sizes: [d_in, hidden1, hidden2, ..., d_out]
        lr: learning rate
        n_epochs: training epochs
        batch_size: mini-batch size
        seed: random seed

    Returns:
        weights: trained weight matrices
        biases: trained bias vectors
        losses: loss at each epoch
    """
    raise NotImplementedError
