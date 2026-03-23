"""Bayesian Optimization Exercises. Reference: Snoek et al. 2012 (arXiv:1206.2944)."""

import torch
import numpy as np


def expected_improvement(mu: torch.Tensor, sigma: torch.Tensor, best_so_far: float) -> torch.Tensor:
    """EI = (mu - f_best) * Phi(z) + sigma * phi(z), where z = (mu - f_best) / sigma."""
    raise NotImplementedError


def upper_confidence_bound(mu: torch.Tensor, sigma: torch.Tensor, beta: float = 2.0) -> torch.Tensor:
    """UCB = mu + beta * sigma."""
    raise NotImplementedError


def probability_of_improvement(mu: torch.Tensor, sigma: torch.Tensor, best_so_far: float) -> torch.Tensor:
    """PI = Phi((mu - f_best) / sigma)."""
    raise NotImplementedError


def bayesian_optimization_loop(objective: callable, bounds: np.ndarray, n_iterations: int, acquisition_fn: callable) -> dict:
    """Full BO loop with GP surrogate."""
    raise NotImplementedError


def thompson_sampling(gp_posterior: callable, n_candidates: int) -> torch.Tensor:
    """Sample from GP posterior, maximize."""
    raise NotImplementedError
