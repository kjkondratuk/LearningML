"""
Mini-Project: Kernel SVM on Non-Linear Data
=============================================

Spiral 2, Phase 2B, Module 05

Generate three datasets: linearly separable, concentric circles, two moons.
Train linear, polynomial, and RBF kernel SVMs on each (9 combos).
Visualize decision boundaries and support vectors.
"""

import numpy as np


def generate_linearly_separable(n: int = 200, seed: int = 42):
    """Two Gaussian blobs that are linearly separable."""
    raise NotImplementedError


def generate_concentric_circles(n: int = 200, seed: int = 42):
    """Two concentric circles (inner and outer)."""
    raise NotImplementedError


def generate_two_moons(n: int = 200, noise: float = 0.1, seed: int = 42):
    """Two interleaving half-circles."""
    raise NotImplementedError


def compute_decision_boundary(
    predict_fn: callable,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    resolution: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute decision boundary on a grid for visualization.

    Returns:
        X_grid, Y_grid, Z_grid for contour plotting
    """
    raise NotImplementedError
