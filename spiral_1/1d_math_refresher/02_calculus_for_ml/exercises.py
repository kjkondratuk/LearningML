"""
Module 02: Calculus for ML
============================
Derivatives, gradients, and the chain rule -- the engine of training.
"""

import numpy as np
from typing import Callable


def numerical_derivative(f: Callable[[float], float], x: float,
                         h: float = 1e-7) -> float:
    """Compute the numerical derivative of f at x using central differences.

    f'(x) ~ (f(x+h) - f(x-h)) / (2h)

    Args:
        f: A scalar function of one variable.
        x: The point at which to evaluate the derivative.
        h: Step size.

    Returns:
        Approximate derivative (float).
    """
    raise NotImplementedError


def numerical_partial_derivatives(
    f: Callable[[np.ndarray], float], x: np.ndarray, h: float = 1e-7
) -> np.ndarray:
    """Compute the gradient of f at x using numerical partial derivatives.

    For each dimension i, perturb x[i] by +h and -h and use central differences.

    Args:
        f: A scalar function of a vector.
        x: 1D array, the point at which to compute the gradient.
        h: Step size.

    Returns:
        1D array of partial derivatives (the gradient), same shape as x.
    """
    raise NotImplementedError


def gradient_of_mse(X: np.ndarray, y: np.ndarray,
                    w: np.ndarray) -> np.ndarray:
    """Compute the analytical gradient of MSE loss w.r.t. weights w.

    MSE = (1/n) * ||X @ w - y||^2
    dMSE/dw = (2/n) * X.T @ (X @ w - y)

    Args:
        X: Design matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples,).
        w: Weight vector of shape (n_features,).

    Returns:
        Gradient vector of shape (n_features,).
    """
    raise NotImplementedError


def chain_rule_demo(x: float) -> dict[str, float]:
    """Demonstrate the chain rule on f(x) = sin(x^2).

    Let u = x^2, f = sin(u).
    du/dx = 2x
    df/du = cos(u)
    df/dx = cos(x^2) * 2x

    Args:
        x: The point at which to evaluate.

    Returns:
        Dict with keys:
        - "f": f(x) = sin(x^2)
        - "df_dx_analytical": cos(x^2) * 2x
        - "df_dx_numerical": numerical derivative of sin(x^2)
    """
    raise NotImplementedError


def gradient_descent_1d(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> tuple[float, list[float]]:
    """Run gradient descent on a 1D function.

    x_{t+1} = x_t - lr * df(x_t)

    Stop when |df(x_t)| < tol or max_iter is reached.

    Args:
        f: The objective function.
        df: The derivative of f.
        x0: Starting point.
        lr: Learning rate.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on gradient magnitude.

    Returns:
        Tuple of (x_final, history) where history is a list of x values.
    """
    raise NotImplementedError


def gradient_descent_2d(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-8,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Run gradient descent on a 2D function.

    x_{t+1} = x_t - lr * grad_f(x_t)

    Stop when ||grad_f(x_t)|| < tol or max_iter is reached.

    Args:
        f: Objective function taking a 2D array.
        grad_f: Gradient function returning a 2D array.
        x0: Starting point, shape (2,).
        lr: Learning rate.
        max_iter: Maximum iterations.
        tol: Convergence tolerance on gradient norm.

    Returns:
        Tuple of (x_final, history) where history is a list of 2D arrays.
    """
    raise NotImplementedError
