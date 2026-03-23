"""
Measure-Theoretic Probability Exercises

Reference: Pollard, "A User's Guide to Measure Theoretic Probability" (2001)
"""

import torch
from typing import Callable


def lebesgue_integral_mc(
    f: Callable, domain: tuple[float, float], n_samples: int = 10000
) -> tuple[float, float]:
    """Approximate a Lebesgue integral via Monte Carlo.

    integral_a^b f(x) dx ~ (b-a) * (1/n) * sum f(x_i), x_i ~ Uniform(a,b)

    Args:
        f: Function to integrate.
        domain: (a, b) integration bounds.
        n_samples: Number of MC samples.

    Returns:
        Tuple of (estimate, standard_error).
    """
    raise NotImplementedError


def importance_sampling(
    f: Callable,
    target_density: Callable,
    proposal_density: Callable,
    proposal_sampler: Callable,
    n_samples: int = 10000,
) -> tuple[float, float]:
    """Importance sampling estimate of E_p[f(x)].

    E_p[f] = E_q[f(x) * p(x) / q(x)]

    Args:
        f: Function to compute expectation of.
        target_density: p(x), the target density.
        proposal_density: q(x), the proposal density.
        proposal_sampler: Function that returns n samples from q.
        n_samples: Number of samples.

    Returns:
        Tuple of (estimate, variance_of_estimate).
    """
    raise NotImplementedError


def radon_nikodym_discrete(
    p: torch.Tensor, q: torch.Tensor
) -> torch.Tensor:
    """Compute density ratio dP/dQ for discrete distributions.

    (dP/dQ)(x) = P(x) / Q(x)

    Args:
        p: Probability mass function of P, shape (K,).
        q: Probability mass function of Q, shape (K,).

    Returns:
        Density ratio, shape (K,). Inf where Q(x) = 0 but P(x) > 0.
    """
    raise NotImplementedError


def change_of_variables(
    f_inv: Callable,
    jacobian_det: Callable,
    y: torch.Tensor,
    log_p_x: Callable,
) -> torch.Tensor:
    """Change of variables formula for normalizing flows.

    log p_Y(y) = log p_X(f^{-1}(y)) + log |det(J_{f^{-1}}(y))|

    Args:
        f_inv: Inverse transformation x = f^{-1}(y).
        jacobian_det: |det(J_{f^{-1}}(y))| for given y.
        y: Points in Y space, shape (n, d).
        log_p_x: Log density of base distribution p_X.

    Returns:
        Log density log p_Y(y), shape (n,).
    """
    raise NotImplementedError


def kl_divergence_mc(
    log_p: Callable,
    log_q: Callable,
    samples_from_p: torch.Tensor,
) -> torch.Tensor:
    """Monte Carlo estimate of KL(P || Q).

    KL(P || Q) = E_P[log P(x) - log Q(x)] ~ (1/n) sum [log p(x_i) - log q(x_i)]

    Args:
        log_p: Log density of P.
        log_q: Log density of Q.
        samples_from_p: Samples from P, shape (n, d).

    Returns:
        KL divergence estimate (scalar).
    """
    raise NotImplementedError
