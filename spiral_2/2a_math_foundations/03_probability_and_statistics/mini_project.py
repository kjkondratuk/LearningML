"""
Mini-Project: Bayesian Coin-Flip Inference
===========================================

Spiral 2, Phase 2A, Module 03

Watch a Beta prior evolve as coin flips arrive one at a time:
1. Start with Beta(1,1) -- uniform prior
2. After each flip, update the posterior
3. Animate or plot the posterior at each step
4. Compare posterior mean vs MLE vs MAP
5. Show how a strong prior Beta(100,100) resists evidence from a biased coin
"""

import numpy as np


def beta_pdf(x: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Compute the Beta distribution PDF.

    p(x) = x^{alpha-1} (1-x)^{beta-1} / B(alpha, beta)

    where B(alpha, beta) = Gamma(alpha) Gamma(beta) / Gamma(alpha + beta).

    Args:
        x: points in [0, 1]
        alpha: first shape parameter
        beta: second shape parameter

    Returns:
        pdf values, same shape as x
    """
    raise NotImplementedError


def sequential_bayesian_update(
    flips: list[int],
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
) -> list[tuple[float, float]]:
    """Update Beta posterior after each coin flip.

    Args:
        flips: list of 0 (tails) and 1 (heads)
        prior_alpha: initial alpha
        prior_beta: initial beta

    Returns:
        list of (alpha, beta) tuples, one per flip (length = len(flips))
    """
    raise NotImplementedError


def posterior_mean(alpha: float, beta: float) -> float:
    """Mean of Beta(alpha, beta) = alpha / (alpha + beta)."""
    raise NotImplementedError


def posterior_mode(alpha: float, beta: float) -> float:
    """Mode (MAP) of Beta(alpha, beta) = (alpha - 1) / (alpha + beta - 2).

    Only valid when alpha > 1 and beta > 1.
    """
    raise NotImplementedError


def mle_estimate(flips: list[int]) -> float:
    """MLE estimate = number of heads / total flips.

    Args:
        flips: list of 0 and 1 values

    Returns:
        MLE estimate of p(heads)
    """
    raise NotImplementedError


def strong_prior_resistance(
    true_p: float,
    n_flips: int,
    strong_alpha: float = 100.0,
    strong_beta: float = 100.0,
    seed: int = 42,
) -> dict:
    """Demonstrate how a strong prior resists evidence.

    Generate n_flips from Bernoulli(true_p) and track the posterior
    under both a weak prior Beta(1,1) and a strong prior Beta(100,100).

    Args:
        true_p: true coin bias
        n_flips: number of flips to simulate
        strong_alpha, strong_beta: strong prior parameters
        seed: random seed

    Returns:
        dict with 'weak_posteriors' and 'strong_posteriors' keys,
        each a list of (alpha, beta) tuples
    """
    raise NotImplementedError
