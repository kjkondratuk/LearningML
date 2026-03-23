"""
CNN Exercises -- Spiral 2, Phase 2C, Module 05
"""

import numpy as np


def conv2d_naive(
    X: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0
) -> np.ndarray:
    """2D convolution with nested loops (the slow way).

    Args:
        X: shape (batch, C_in, H, W)
        kernel: shape (C_out, C_in, kH, kW)
        stride: stride in both dimensions
        padding: zero-padding on each side

    Returns:
        output: shape (batch, C_out, H_out, W_out)
    """
    raise NotImplementedError


def conv2d_im2col(
    X: np.ndarray, kernel: np.ndarray, stride: int = 1, padding: int = 0
) -> np.ndarray:
    """2D convolution via im2col trick.

    Reshape input patches into columns, then conv = matrix multiply.

    Args:
        X: shape (batch, C_in, H, W)
        kernel: shape (C_out, C_in, kH, kW)
        stride, padding: as usual

    Returns:
        output: shape (batch, C_out, H_out, W_out)
    """
    raise NotImplementedError


def conv2d_backward(
    dout: np.ndarray, X: np.ndarray, kernel: np.ndarray,
    stride: int = 1, padding: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Backward pass for 2D convolution.

    Derive: dL/d_kernel is also a convolution.
    dL/d_input is a "full" convolution with the flipped kernel.

    Returns:
        dX: gradient w.r.t. input
        dkernel: gradient w.r.t. kernel
    """
    raise NotImplementedError


def max_pool_forward(
    X: np.ndarray, pool_size: int = 2, stride: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Max pooling forward.

    Store argmax positions for backward.

    Args:
        X: shape (batch, C, H, W)
        pool_size: size of pooling window
        stride: stride

    Returns:
        out: pooled output
        cache: argmax positions for backward
    """
    raise NotImplementedError


def max_pool_backward(
    dout: np.ndarray, cache: np.ndarray
) -> np.ndarray:
    """Route gradients only to the max positions."""
    raise NotImplementedError


def receptive_field_calculator(
    layer_configs: list[tuple[int, int, int]]
) -> list[int]:
    """Compute receptive field at each layer.

    Args:
        layer_configs: list of (kernel_size, stride, padding) tuples

    Returns:
        receptive_fields: list of RF sizes, one per layer
    """
    raise NotImplementedError


def build_cnn(
    input_shape: tuple,
    conv_configs: list[dict],
    fc_sizes: list[int],
    n_classes: int,
    seed: int = 42,
) -> dict:
    """Assemble a CNN from conv layers, pooling, and FC layers.

    Returns a model dict with all weights/biases.
    """
    raise NotImplementedError


def forward_cnn(X: np.ndarray, model: dict) -> tuple[np.ndarray, list]:
    """Forward pass through the full CNN."""
    raise NotImplementedError


def backward_cnn(loss_grad: np.ndarray, model: dict, caches: list) -> dict:
    """Full backward pass through the CNN."""
    raise NotImplementedError
