"""
Activation Functions and Weight Initialization -- Spiral 2, Phase 2C, Module 02
"""

import numpy as np
from typing import Callable


def sigmoid_and_gradient(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sigmoid and its gradient.

    sigma(z) = 1/(1+exp(-z))
    sigma'(z) = sigma(z)(1-sigma(z))

    Vanishing gradient: sigma' -> 0 for |z| > 5.

    Returns: (activation, gradient)
    """
    raise NotImplementedError


def tanh_and_gradient(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Tanh and its gradient.

    tanh(z) = (exp(z) - exp(-z)) / (exp(z) + exp(-z))
    tanh'(z) = 1 - tanh(z)^2

    Rescaled sigmoid: tanh(z) = 2*sigma(2z) - 1. Still vanishes.

    Returns: (activation, gradient)
    """
    raise NotImplementedError


def relu_and_gradient(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ReLU and its subgradient.

    relu(z) = max(0, z)
    relu'(z) = 1 if z > 0, else 0

    No vanishing, but "dying ReLU": neurons with z < 0 always get 0 gradient.

    Returns: (activation, gradient)
    """
    raise NotImplementedError


def leaky_relu_and_gradient(
    z: np.ndarray, alpha: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    """Leaky ReLU fixes dying ReLU by allowing small negative slope.

    leaky_relu(z) = z if z > 0, else alpha*z
    gradient: 1 if z > 0, else alpha

    Returns: (activation, gradient)
    """
    raise NotImplementedError


def gelu_and_gradient(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GELU (Gaussian Error Linear Unit) -- the Transformer activation.

    gelu(z) = z * Phi(z) where Phi is the standard Gaussian CDF.
    Approximate: gelu(z) ~ 0.5*z*(1 + tanh(sqrt(2/pi)*(z + 0.044715*z^3)))

    Returns: (activation, gradient)
    """
    raise NotImplementedError


def xavier_init(fan_in: int, fan_out: int, seed: int = None) -> np.ndarray:
    """Xavier (Glorot) initialization.

    Derive: Var(w) = 2/(fan_in + fan_out).
    Sample from N(0, 2/(fan_in + fan_out)) or U(-a, a) with a = sqrt(6/(fan_in+fan_out)).

    Maintains activation variance through layers with sigmoid/tanh.

    Args:
        fan_in: number of input units
        fan_out: number of output units
        seed: optional random seed

    Returns:
        W: shape (fan_in, fan_out)
    """
    raise NotImplementedError


def he_init(fan_in: int, fan_out: int, seed: int = None) -> np.ndarray:
    """He (Kaiming) initialization for ReLU networks.

    Derive: Var(w) = 2/fan_in (ReLU zeros half the outputs).
    Sample from N(0, 2/fan_in).

    Args:
        fan_in: number of input units
        fan_out: number of output units
        seed: optional random seed

    Returns:
        W: shape (fan_in, fan_out)
    """
    raise NotImplementedError


def track_activation_statistics(
    X: np.ndarray,
    weights_list: list[np.ndarray],
    activation_fn: Callable,
) -> list[tuple[float, float]]:
    """Forward-pass X through a deep network, recording activation statistics.

    At each layer, record mean and variance of activations.
    Use to diagnose vanishing/exploding activations.

    Args:
        X: input data, shape (batch, d)
        weights_list: list of weight matrices for each layer
        activation_fn: activation function to apply at each layer

    Returns:
        stats: list of (mean, variance) tuples, one per layer
    """
    raise NotImplementedError
