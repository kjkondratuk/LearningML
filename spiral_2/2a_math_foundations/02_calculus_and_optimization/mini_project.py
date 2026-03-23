"""
Mini-Project: Gradient Descent Visualizer
==========================================

Spiral 2, Phase 2A, Module 02

Create contour plots for 3 test functions and overlay GD trajectories:
1. Quadratic bowl (well-conditioned and ill-conditioned)
2. Rosenbrock banana function
3. Saddle point function

For each, show:
- GD with different learning rates
- GD with momentum vs without
- A divergence case (lr too high)
"""

import numpy as np


def quadratic_bowl(x: np.ndarray, condition_number: float = 1.0) -> float:
    """f(x,y) = 0.5 * (x^2 + condition_number * y^2).

    Minimum at origin. Ill-conditioned when condition_number >> 1.
    """
    raise NotImplementedError


def quadratic_bowl_grad(x: np.ndarray, condition_number: float = 1.0) -> np.ndarray:
    """Gradient of the quadratic bowl."""
    raise NotImplementedError


def rosenbrock(x: np.ndarray) -> float:
    """f(x,y) = (1-x)^2 + 100*(y-x^2)^2.

    Famous banana-shaped valley. Minimum at (1, 1).
    """
    raise NotImplementedError


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of the Rosenbrock function."""
    raise NotImplementedError


def saddle_point(x: np.ndarray) -> float:
    """f(x,y) = x^2 - y^2.

    Saddle point at origin: minimum along x, maximum along y.
    """
    raise NotImplementedError


def saddle_point_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of the saddle point function."""
    raise NotImplementedError


def create_contour_data(
    f: callable,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    resolution: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create meshgrid data for contour plotting.

    Args:
        f: 2D function
        xlim: (xmin, xmax)
        ylim: (ymin, ymax)
        resolution: grid points per axis

    Returns:
        X, Y, Z: meshgrid arrays for contour plotting
    """
    raise NotImplementedError


def run_optimizer_comparison(
    f: callable,
    grad_f: callable,
    x0: np.ndarray,
    lr_values: list[float],
    momentum_values: list[float],
    n_steps: int = 100,
) -> dict:
    """Run GD with various lr and momentum settings, collect trajectories.

    Returns:
        dict mapping (lr, momentum) -> trajectory (list of np.ndarray)
    """
    raise NotImplementedError
