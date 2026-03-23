"""
Regularization for Deep Learning -- Spiral 2, Phase 2C, Module 04
"""

import numpy as np
from typing import Optional


def dropout_forward(
    X: np.ndarray, p: float, training: bool, seed: int = None
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Dropout: randomly zero neurons during training.

    During training: mask each neuron with probability p, scale by 1/(1-p).
    During inference: do nothing (identity).

    The 1/(1-p) scaling ensures E[output] is the same at train and test time.

    Args:
        X: input activations
        p: dropout probability (fraction to zero out)
        training: True during training, False at inference
        seed: optional random seed

    Returns:
        output: same shape as X
        mask: binary mask used (None if not training)
    """
    raise NotImplementedError


def dropout_backward(
    dout: np.ndarray, mask: np.ndarray, p: float
) -> np.ndarray:
    """Backprop through dropout: gradient flows only through kept neurons."""
    raise NotImplementedError


def batch_norm_forward(
    X: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    training: bool,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> tuple[np.ndarray, dict]:
    """Batch normalization forward pass.

    Training: normalize per feature using batch statistics.
    Inference: use running mean/var.

    x_hat = (x - mu_batch) / sqrt(var_batch + eps)
    y = gamma * x_hat + beta

    Args:
        X: shape (batch, d)
        gamma: shape (d,), scale parameter
        beta: shape (d,), shift parameter
        running_mean, running_var: shape (d,), running statistics
        training: True for training mode
        momentum: for running statistics update
        eps: numerical stability

    Returns:
        out: shape (batch, d)
        cache: dict of intermediate values for backward
    """
    raise NotImplementedError


def batch_norm_backward(dout: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward pass for batch normalization.

    This is notoriously tricky -- derive step by step.

    Returns:
        dx: gradient w.r.t. input
        dgamma: gradient w.r.t. gamma
        dbeta: gradient w.r.t. beta
    """
    raise NotImplementedError


def layer_norm_forward(
    X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> tuple[np.ndarray, dict]:
    """Layer normalization: normalize per sample (not per feature).

    Transformers use LayerNorm instead of BatchNorm because:
    1. It works with variable-length sequences
    2. It does not depend on batch size
    3. It normalizes across features, not across the batch

    Args:
        X: shape (batch, d)
        gamma, beta: shape (d,)
        eps: numerical stability

    Returns:
        out: shape (batch, d)
        cache: for backward
    """
    raise NotImplementedError


def layer_norm_backward(dout: np.ndarray, cache: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Backward pass for layer normalization.

    Returns: (dx, dgamma, dbeta)
    """
    raise NotImplementedError


def weight_decay_vs_l2(
    params: np.ndarray,
    gradients: np.ndarray,
    weight_decay: float,
    l2_lambda: float,
    lr: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Show weight decay and L2 regularization differ for Adam.

    For SGD: both are equivalent.
    For Adam: L2 adds penalty to gradient (scaled by adaptive lr),
              weight decay subtracts from params directly (not scaled).

    Returns:
        params_wd: params updated with weight decay
        params_l2: params updated with L2 regularization
    """
    raise NotImplementedError


def data_augmentation_effect(
    model_fn: callable,
    X: np.ndarray,
    y: np.ndarray,
    augment_fn: callable,
    n_epochs: int = 50,
) -> tuple[list[float], list[float]]:
    """Train with and without augmentation, compare.

    Returns:
        losses_no_aug: training losses without augmentation
        losses_aug: training losses with augmentation
    """
    raise NotImplementedError
