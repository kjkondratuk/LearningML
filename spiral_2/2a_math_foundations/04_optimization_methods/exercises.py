"""
Optimization Methods Exercises -- Spiral 2, Phase 2A, Module 04

Implement the full family of gradient-based optimizers from scratch.
Each optimizer accepts a batch_fn for stochastic gradients.
"""

import numpy as np
from typing import Callable, Optional


def sgd(
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float,
    n_steps: int,
    batch_fn: Optional[Callable[[], None]] = None,
) -> list[np.ndarray]:
    """Stochastic gradient descent with mini-batch gradient function.

    Update: x_{t+1} = x_t - lr * grad_f(x_t)

    If batch_fn is provided, call it before each gradient evaluation
    to resample the mini-batch.

    Args:
        grad_f: gradient function (may use current mini-batch internally)
        x0: starting point
        lr: learning rate
        n_steps: number of iterations
        batch_fn: optional callable to resample the mini-batch

    Returns:
        trajectory: list of x values (length n_steps + 1)
    """
    raise NotImplementedError


def sgd_momentum(
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float,
    momentum: float,
    n_steps: int,
    batch_fn: Optional[Callable[[], None]] = None,
) -> list[np.ndarray]:
    """SGD with classical (heavy-ball) momentum.

    v_{t+1} = momentum * v_t + grad_f(x_t)
    x_{t+1} = x_t - lr * v_{t+1}

    Args:
        grad_f: gradient function
        x0: starting point
        lr: learning rate
        momentum: momentum coefficient (typically 0.9)
        n_steps: number of iterations
        batch_fn: optional mini-batch resampler

    Returns:
        trajectory: list of x values
    """
    raise NotImplementedError


def nesterov_momentum(
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float,
    momentum: float,
    n_steps: int,
    batch_fn: Optional[Callable[[], None]] = None,
) -> list[np.ndarray]:
    """Nesterov accelerated gradient.

    Derive the "look-ahead" step:
    v_{t+1} = momentum * v_t + grad_f(x_t - lr * momentum * v_t)
    x_{t+1} = x_t - lr * v_{t+1}

    The gradient is evaluated at the look-ahead point, not the current point.

    Args:
        grad_f: gradient function
        x0: starting point
        lr: learning rate
        momentum: momentum coefficient
        n_steps: number of iterations
        batch_fn: optional mini-batch resampler

    Returns:
        trajectory: list of x values
    """
    raise NotImplementedError


def adagrad(
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float,
    n_steps: int,
    batch_fn: Optional[Callable[[], None]] = None,
    epsilon: float = 1e-8,
) -> list[np.ndarray]:
    """Adaptive Gradient (AdaGrad).

    Derive from the idea: parameters that rarely update should have
    larger learning rates.

    G_t = G_{t-1} + grad_t^2   (element-wise squared gradient accumulator)
    x_{t+1} = x_t - (lr / sqrt(G_t + epsilon)) * grad_t

    Note: G_t only grows -- the effective learning rate only decreases.

    Args:
        grad_f: gradient function
        x0: starting point
        lr: base learning rate
        n_steps: number of iterations
        batch_fn: optional mini-batch resampler
        epsilon: small constant for numerical stability

    Returns:
        trajectory: list of x values
    """
    raise NotImplementedError


def rmsprop(
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float,
    decay: float,
    n_steps: int,
    batch_fn: Optional[Callable[[], None]] = None,
    epsilon: float = 1e-8,
) -> list[np.ndarray]:
    """RMSProp: fixes AdaGrad's monotonically decreasing learning rate.

    Uses an exponential moving average of squared gradients instead of sum:
    v_t = decay * v_{t-1} + (1 - decay) * grad_t^2
    x_{t+1} = x_t - (lr / sqrt(v_t + epsilon)) * grad_t

    Args:
        grad_f: gradient function
        x0: starting point
        lr: learning rate
        decay: decay rate for moving average (typically 0.9)
        n_steps: number of iterations
        batch_fn: optional mini-batch resampler
        epsilon: numerical stability constant

    Returns:
        trajectory: list of x values
    """
    raise NotImplementedError


def adam(
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float,
    beta1: float,
    beta2: float,
    n_steps: int,
    batch_fn: Optional[Callable[[], None]] = None,
    epsilon: float = 1e-8,
) -> list[np.ndarray]:
    """Adam: Adaptive Moment Estimation.

    Derive as momentum + RMSProp + bias correction:
    m_t = beta1 * m_{t-1} + (1 - beta1) * grad_t          (first moment)
    v_t = beta2 * v_{t-1} + (1 - beta2) * grad_t^2        (second moment)
    m_hat = m_t / (1 - beta1^t)                            (bias correction)
    v_hat = v_t / (1 - beta2^t)                            (bias correction)
    x_{t+1} = x_t - lr * m_hat / (sqrt(v_hat) + epsilon)

    Args:
        grad_f: gradient function
        x0: starting point
        lr: learning rate (default 0.001 in the paper)
        beta1: first moment decay (default 0.9)
        beta2: second moment decay (default 0.999)
        n_steps: number of iterations
        batch_fn: optional mini-batch resampler
        epsilon: numerical stability constant

    Returns:
        trajectory: list of x values
    """
    raise NotImplementedError


def lbfgs_step(
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    history: list[tuple[np.ndarray, np.ndarray]],
    m: int = 10,
) -> np.ndarray:
    """Compute one L-BFGS step using the two-loop recursion.

    L-BFGS approximates the inverse Hessian using the last m
    (position_change, gradient_change) pairs.

    The two-loop recursion computes H_k * grad without forming H_k.

    Explain why second-order methods are impractical for deep learning
    (Hessian is n^2 for n parameters) but great for convex problems.

    Args:
        grad_f: gradient function
        x: current point
        history: list of (s_k, y_k) pairs where s_k = x_{k+1} - x_k,
                 y_k = grad_{k+1} - grad_k
        m: number of history pairs to keep

    Returns:
        direction: the search direction -H_k * grad_f(x)
    """
    raise NotImplementedError


def line_search(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    direction: np.ndarray,
    alpha: float = 1.0,
    c: float = 1e-4,
    rho: float = 0.9,
) -> float:
    """Backtracking line search with the Armijo condition.

    Find step size alpha such that:
        f(x + alpha * d) <= f(x) + c * alpha * grad_f(x)^T d

    Start with alpha and reduce by factor rho until condition is met.

    Args:
        f: objective function
        grad_f: gradient function
        x: current point
        direction: search direction
        alpha: initial step size
        c: Armijo condition constant (typically 1e-4)
        rho: backtracking factor (typically 0.9)

    Returns:
        alpha: the step size satisfying the Armijo condition
    """
    raise NotImplementedError
