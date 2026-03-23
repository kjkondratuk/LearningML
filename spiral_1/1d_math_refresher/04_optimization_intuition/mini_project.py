"""
Mini-Project: Visualize Optimization Landscape
================================================
Plot optimization trajectories on 2D loss surfaces to build intuition
about how different optimizers behave.
"""

import numpy as np


def rosenbrock(x: np.ndarray) -> float:
    """The Rosenbrock function: f(x,y) = (1-x)^2 + 100*(y - x^2)^2.

    A classic test function with a narrow curved valley.
    Minimum at (1, 1) with f(1,1) = 0.

    Args:
        x: Array of shape (2,), representing (x, y).

    Returns:
        Function value (scalar).
    """
    raise NotImplementedError


def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
    """Gradient of the Rosenbrock function.

    df/dx = -2*(1-x) - 400*x*(y - x^2)
    df/dy = 200*(y - x^2)

    Args:
        x: Array of shape (2,).

    Returns:
        Gradient array of shape (2,).
    """
    raise NotImplementedError


def beale(x: np.ndarray) -> float:
    """The Beale function -- another classic 2D test surface.

    f(x,y) = (1.5 - x + x*y)^2 + (2.25 - x + x*y^2)^2 + (2.625 - x + x*y^3)^2
    Minimum at (3, 0.5) with f = 0.

    Args:
        x: Array of shape (2,).

    Returns:
        Function value (scalar).
    """
    raise NotImplementedError


def beale_grad(x: np.ndarray) -> np.ndarray:
    """Numerical gradient of the Beale function.

    Uses central differences for simplicity.

    Args:
        x: Array of shape (2,).

    Returns:
        Gradient array of shape (2,).
    """
    raise NotImplementedError


def visualize_optimization_landscape(
    func_name: str = "rosenbrock",
    methods: list[str] | None = None,
    lr: float = 0.001,
    max_iter: int = 5000,
    save_path: str | None = None,
) -> dict:
    """Generate optimization trajectories and (optionally) plot them.

    Args:
        func_name: "rosenbrock" or "beale".
        methods: List of optimizer names, e.g. ["vanilla", "momentum", "adam"].
            Defaults to all three.
        lr: Learning rate.
        max_iter: Maximum iterations.
        save_path: If provided, save the plot to this path (requires matplotlib).

    Returns:
        Dict mapping method name -> dict with keys:
        - "trajectory": list of (x, y) tuples
        - "final_point": (x, y) tuple
        - "final_loss": float
        - "n_iterations": int
    """
    raise NotImplementedError
