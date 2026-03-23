"""
Calculus and Optimization Exercises -- Spiral 2, Phase 2A, Module 02

All gradient/derivative computations use numerical methods (central differences).
Compare against known analytical results in tests.
"""

import numpy as np
from typing import Callable, Union


def numerical_gradient(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Compute the gradient of f at x using central differences.

    For each component x_i, compute:
        df/dx_i ~ [f(x + e_i * epsilon) - f(x - e_i * epsilon)] / (2 * epsilon)

    Works for scalar input (x is a float or 0-d array) and vector input.

    Args:
        f: scalar-valued function
        x: point at which to evaluate the gradient; shape (n,) or scalar
        epsilon: step size for finite differences

    Returns:
        grad: same shape as x
    """
    raise NotImplementedError


def numerical_jacobian(
    f: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Compute the Jacobian matrix of vector-valued f at x.

    J[i, j] = df_i / dx_j, computed via central differences.

    Args:
        f: function mapping R^n -> R^m
        x: shape (n,)
        epsilon: step size

    Returns:
        J: shape (m, n)
    """
    raise NotImplementedError


def numerical_hessian(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """Compute the Hessian matrix of scalar-valued f at x.

    H[i, j] = d^2f / dx_i dx_j

    Compute by applying numerical_gradient to each component of the gradient,
    or use the second-order central difference formula directly.

    The result should be symmetric (verify in tests).

    Args:
        f: scalar-valued function
        x: shape (n,)
        epsilon: step size

    Returns:
        H: shape (n, n), symmetric
    """
    raise NotImplementedError


def taylor_approximate(
    f: Callable[[np.ndarray], float],
    x0: np.ndarray,
    order: int,
    points: np.ndarray,
) -> np.ndarray:
    """Compute Taylor approximation of f around x0 at given points.

    Order 1: f(x) ~ f(x0) + grad_f(x0)^T (x - x0)
    Order 2: f(x) ~ f(x0) + grad^T dx + 0.5 dx^T H dx

    Args:
        f: scalar-valued function
        x0: expansion point, shape (n,)
        order: 1 or 2
        points: shape (k, n) -- points at which to evaluate the approximation

    Returns:
        values: shape (k,) -- Taylor approximation at each point
    """
    raise NotImplementedError


def gradient_descent(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float,
    n_steps: int,
) -> list[np.ndarray]:
    """Basic gradient descent.

    Update rule: x_{t+1} = x_t - lr * grad_f(x_t)

    Args:
        f: objective function (for logging, not used in update)
        grad_f: gradient function
        x0: starting point
        lr: learning rate
        n_steps: number of iterations

    Returns:
        trajectory: list of n_steps+1 arrays (including x0)
    """
    raise NotImplementedError


def gradient_descent_with_momentum(
    f: Callable[[np.ndarray], float],
    grad_f: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    lr: float,
    momentum: float,
    n_steps: int,
) -> list[np.ndarray]:
    """Gradient descent with classical (heavy-ball) momentum.

    Update rule:
        v_{t+1} = momentum * v_t + grad_f(x_t)
        x_{t+1} = x_t - lr * v_{t+1}

    Args:
        f: objective function
        grad_f: gradient function
        x0: starting point
        lr: learning rate
        momentum: momentum coefficient (e.g., 0.9)
        n_steps: number of iterations

    Returns:
        trajectory: list of n_steps+1 arrays (including x0)
    """
    raise NotImplementedError


def check_convexity(
    f: Callable[[np.ndarray], float],
    x_range: tuple[np.ndarray, np.ndarray],
    n_samples: int = 100,
) -> bool:
    """Numerically verify convexity of f over a region.

    For n_samples random pairs (x, y) in the region, check:
        f(t*x + (1-t)*y) <= t*f(x) + (1-t)*f(y)
    for several values of t in (0, 1).

    Args:
        f: scalar-valued function
        x_range: (lower_bound, upper_bound) defining the box region
        n_samples: number of random pairs to test

    Returns:
        True if all checks pass (f appears convex), False otherwise
    """
    raise NotImplementedError


def lagrange_dual(
    f: Callable[[np.ndarray], float],
    g: Callable[[np.ndarray], float],
    x_range: tuple[float, float],
    n_grid: int = 1000,
) -> tuple[np.ndarray, float]:
    """Solve min f(x) subject to g(x) = 0 using Lagrange multipliers (2D case).

    The Lagrangian is L(x, lambda) = f(x) + lambda * g(x).
    At the solution: grad_f + lambda * grad_g = 0 and g(x) = 0.

    For the 2D case, use a grid search over x_range to find the constrained
    minimum, then verify the KKT conditions.

    Args:
        f: objective function (2D input)
        g: equality constraint function (g(x) = 0)
        x_range: (min_val, max_val) for each dimension
        n_grid: grid resolution per dimension

    Returns:
        x_star: the optimal point, shape (2,)
        lambda_star: the Lagrange multiplier
    """
    raise NotImplementedError
