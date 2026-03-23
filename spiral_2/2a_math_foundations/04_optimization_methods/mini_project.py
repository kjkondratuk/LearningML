"""
Mini-Project: Optimizer Shootout
=================================

Spiral 2, Phase 2A, Module 04

Compare all optimizers on 4 test surfaces:
1. Convex quadratic (well-conditioned and ill-conditioned)
2. Rosenbrock banana function
3. Rastrigin (non-convex, many local minima)
4. Logistic regression loss on a small dataset

Produce:
- Convergence curves (loss vs iteration) for each optimizer
- Trajectory plots on contour plots (for 2D surfaces)
- Comparison table: steps to convergence, final loss, wall-clock time
- 1-paragraph analysis of when to use each optimizer
"""

import numpy as np
from typing import Callable


def convex_quadratic(x: np.ndarray, condition: float = 1.0) -> float:
    """f(x) = 0.5 * (x[0]^2 + condition * x[1]^2)."""
    raise NotImplementedError


def convex_quadratic_grad(x: np.ndarray, condition: float = 1.0) -> np.ndarray:
    """Gradient of convex_quadratic."""
    raise NotImplementedError


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin function: many local minima, global minimum at origin.

    f(x) = 10*d + sum(x_i^2 - 10*cos(2*pi*x_i))
    """
    raise NotImplementedError


def rastrigin_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of the Rastrigin function."""
    raise NotImplementedError


def logistic_loss(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Binary logistic regression negative log-likelihood.

    NLL = -sum(y * log(sigmoid(Xw)) + (1-y) * log(1-sigmoid(Xw)))
    """
    raise NotImplementedError


def logistic_grad(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gradient of logistic loss."""
    raise NotImplementedError


def run_shootout(
    optimizers: dict[str, Callable],
    f: Callable,
    grad_f: Callable,
    x0: np.ndarray,
    n_steps: int = 500,
) -> dict[str, dict]:
    """Run all optimizers and collect results.

    Returns:
        dict mapping optimizer_name -> {
            'trajectory': list[np.ndarray],
            'losses': list[float],
            'time': float (seconds),
            'final_loss': float,
            'steps_to_convergence': int or None,
        }
    """
    raise NotImplementedError


def generate_comparison_table(results: dict[str, dict]) -> str:
    """Generate a markdown table comparing optimizer performance.

    Columns: Optimizer | Final Loss | Steps to Conv | Wall Time
    """
    raise NotImplementedError
