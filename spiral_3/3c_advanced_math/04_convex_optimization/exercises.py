"""
Convex Optimization Exercises

Reference: Boyd & Vandenberghe, "Convex Optimization" (2004)
"""

import torch
import numpy as np
from typing import Callable


def verify_convexity(f: Callable, domain_points: torch.Tensor) -> bool:
    """Check convexity: f(lambda*x + (1-lambda)*y) <= lambda*f(x) + (1-lambda)*f(y).

    Tests for multiple random pairs and lambda values.

    Args:
        f: Function to test (takes tensor, returns scalar).
        domain_points: Points in the domain to test, shape (n, d).

    Returns:
        True if f appears convex on the sampled points.
    """
    raise NotImplementedError


def lagrangian(
    f: Callable, constraints: list[Callable], multipliers: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Form the Lagrangian L(x, lambda) = f(x) + sum lambda_i * g_i(x).

    Args:
        f: Objective function.
        constraints: List of constraint functions g_i(x) <= 0.
        multipliers: Lagrange multipliers, shape (num_constraints,).
        x: Point to evaluate at.

    Returns:
        Lagrangian value (scalar).
    """
    raise NotImplementedError


def kkt_conditions_check(
    x_star: torch.Tensor,
    lambda_star: torch.Tensor,
    f_grad: torch.Tensor,
    g: list[torch.Tensor],
    g_grad: list[torch.Tensor],
    tol: float = 1e-5,
) -> dict[str, bool]:
    """Verify all four KKT conditions.

    1. Stationarity: grad f + sum lambda_i grad g_i = 0
    2. Primal feasibility: g_i(x*) <= 0
    3. Dual feasibility: lambda_i >= 0
    4. Complementary slackness: lambda_i * g_i(x*) = 0

    Returns:
        Dict with keys 'stationarity', 'primal_feasibility',
        'dual_feasibility', 'complementary_slackness'.
    """
    raise NotImplementedError


def proximal_operator_l1(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Soft-thresholding (proximal operator of L1 norm).

    prox_{t*||.||_1}(x) = sign(x) * max(|x| - t, 0)

    Args:
        x: Input tensor.
        threshold: Thresholding parameter t.

    Returns:
        Soft-thresholded tensor.
    """
    raise NotImplementedError


def proximal_gradient_descent(
    f_grad: Callable,
    prox_g: Callable,
    x0: torch.Tensor,
    lr: float,
    iterations: int,
) -> tuple[torch.Tensor, list[float]]:
    """ISTA (Iterative Shrinkage-Thresholding Algorithm) for min f(x) + g(x).

    x_{k+1} = prox_{lr*g}(x_k - lr * grad_f(x_k))

    Args:
        f_grad: Gradient of the smooth part f.
        prox_g: Proximal operator of the non-smooth part g.
        x0: Initial point.
        lr: Step size.
        iterations: Number of iterations.

    Returns:
        Tuple of (final x, list of objective values).
    """
    raise NotImplementedError


def admm_step(
    x: torch.Tensor, z: torch.Tensor, u: torch.Tensor,
    f_prox: Callable, g_prox: Callable, rho: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One step of ADMM for min f(x) + g(z) s.t. x = z.

    x <- prox_{f/rho}(z - u)
    z <- prox_{g/rho}(x + u)
    u <- u + x - z

    Args:
        x, z, u: Current ADMM variables.
        f_prox, g_prox: Proximal operators.
        rho: Augmented Lagrangian parameter.

    Returns:
        Updated (x, z, u).
    """
    raise NotImplementedError


def fenchel_conjugate_quadratic(A: torch.Tensor, b: torch.Tensor) -> Callable:
    """Compute Fenchel conjugate of f(x) = (1/2)x^T A x + b^T x.

    f*(y) = (1/2)(y - b)^T A^{-1} (y - b)

    Args:
        A: PSD matrix, shape (d, d).
        b: Linear term, shape (d,).

    Returns:
        Function that computes f*(y) for a given y.
    """
    raise NotImplementedError
