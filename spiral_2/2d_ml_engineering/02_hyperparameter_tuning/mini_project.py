"""
Mini-Project: Bayesian Optimization from Scratch
==================================================

Spiral 2, Phase 2D, Module 02

Build a full BO library and test on:
1. 1D function (visualize GP posterior and EI at each step)
2. 2D Branin function (standard benchmark)
3. Neural network hyperparameter tuning
"""

import numpy as np


def branin_function(x: np.ndarray) -> float:
    """Branin test function (2D), standard BO benchmark.

    Has 3 global minima, all with value ~0.397887.
    Domain: x1 in [-5, 10], x2 in [0, 15].
    """
    raise NotImplementedError


def visualize_bo_1d(
    objective: callable,
    x_range: tuple[float, float],
    n_iterations: int = 10,
) -> dict:
    """Run BO on 1D function and collect visualization data.

    At each iteration, return:
    - GP posterior mean and uncertainty
    - EI function
    - Next point to evaluate
    - All observed points

    Returns:
        dict with visualization data per iteration
    """
    raise NotImplementedError


def compare_bo_vs_random(
    objective: callable,
    param_space: dict,
    n_evaluations: int = 30,
    n_repeats: int = 10,
    seed: int = 42,
) -> tuple[list[float], list[float]]:
    """Compare sample efficiency of BO vs random search.

    Returns:
        bo_best_per_eval: best value found at each evaluation (averaged)
        random_best_per_eval: same for random search
    """
    raise NotImplementedError
