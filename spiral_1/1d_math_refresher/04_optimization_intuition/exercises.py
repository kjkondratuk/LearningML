"""
Module 04: Optimization Intuition
====================================
The algorithms that train every ML model.
"""

import numpy as np
from typing import Callable


def is_convex_1d(f: Callable[[float], float], x_range: tuple[float, float],
                 n_checks: int = 100) -> bool:
    """Check whether a 1D function is convex on an interval by sampling.

    A function is convex if for all x1, x2 in the interval and t in [0, 1]:
        f(t*x1 + (1-t)*x2) <= t*f(x1) + (1-t)*f(x2)

    Test this with n_checks random pairs of points and t=0.5.

    Args:
        f: A scalar function of one variable.
        x_range: Tuple (x_min, x_max) defining the interval.
        n_checks: Number of random pairs to test.

    Returns:
        True if f appears convex on the interval, False otherwise.
    """
    raise NotImplementedError


def gradient_descent_with_momentum(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float = 0.01,
    momentum: float = 0.9,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Gradient descent with momentum.

    v_t = momentum * v_{t-1} + grad_f(x_t)
    x_{t+1} = x_t - lr * v_t

    Stop when ||grad_f(x_t)|| < tol or max_iter is reached.

    Args:
        f: Objective function.
        grad_f: Gradient function.
        x0: Starting point.
        lr: Learning rate.
        momentum: Momentum coefficient (beta).
        max_iter: Maximum iterations.
        tol: Convergence tolerance.

    Returns:
        Tuple of (x_final, history) where history is a list of x values.
    """
    raise NotImplementedError


def compare_optimizers(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 500,
) -> dict[str, list[float]]:
    """Compare vanilla GD, GD+momentum, and a simple Adam on the same function.

    Return the loss history for each optimizer.

    Adam update rules (simplified):
        m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        m_hat = m_t / (1 - beta1^t)
        v_hat = v_t / (1 - beta2^t)
        x_{t+1} = x_t - lr * m_hat / (sqrt(v_hat) + eps)

    Use beta1=0.9, beta2=0.999, eps=1e-8.

    Args:
        f: Objective function.
        grad_f: Gradient function.
        x0: Starting point (same for all optimizers).
        lr: Learning rate (same for all optimizers).
        max_iter: Maximum iterations.

    Returns:
        Dict with keys "vanilla", "momentum", "adam", each mapping to a list
        of loss values (one per iteration).
    """
    raise NotImplementedError


def learning_rate_schedule(
    schedule_type: str, initial_lr: float, total_steps: int
) -> list[float]:
    """Generate a learning rate schedule.

    Supported types:
        - "constant": lr stays at initial_lr
        - "step": halve lr every total_steps // 3 steps
        - "cosine": cosine annealing from initial_lr to 0

    Cosine formula: lr_t = initial_lr * 0.5 * (1 + cos(pi * t / total_steps))

    Args:
        schedule_type: One of "constant", "step", "cosine".
        initial_lr: Starting learning rate.
        total_steps: Total number of training steps.

    Returns:
        List of learning rates, length total_steps.
    """
    raise NotImplementedError


def find_minimum_2d(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    method: str = "momentum",
    lr: float = 0.01,
    max_iter: int = 2000,
) -> tuple[np.ndarray, float, list[np.ndarray]]:
    """Find the minimum of a 2D function using the specified method.

    Supported methods: "vanilla", "momentum", "adam".

    Args:
        f: Objective function f(x) -> scalar, where x is shape (2,).
        grad_f: Gradient function.
        x0: Starting point, shape (2,).
        method: Optimization method.
        lr: Learning rate.
        max_iter: Maximum iterations.

    Returns:
        Tuple of (x_min, f_min, trajectory) where trajectory is a list of
        x values visited during optimization.
    """
    raise NotImplementedError
