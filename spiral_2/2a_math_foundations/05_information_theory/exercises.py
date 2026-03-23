"""
Information Theory Exercises -- Spiral 2, Phase 2A, Module 05

Implement entropy, cross-entropy, KL divergence, and mutual information.
Convention: 0 * log(0) = 0.
"""

import numpy as np


def entropy(p: np.ndarray) -> float:
    """Compute Shannon entropy of a discrete distribution (in bits).

    H(p) = -sum_i p_i * log2(p_i)

    Handle p_i = 0 by defining 0 * log(0) = 0.

    Args:
        p: shape (K,), probability distribution (sums to 1)

    Returns:
        entropy in bits
    """
    raise NotImplementedError


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """Compute cross-entropy between distributions p and q (in bits).

    H(p, q) = -sum_i p_i * log2(q_i)

    This is the expected number of bits needed to encode samples from p
    using a code optimized for q. Used as the loss function for classification.

    Explain in your implementation: why minimizing H(p, q) w.r.t. q
    makes q approach p.

    Args:
        p: shape (K,), true distribution
        q: shape (K,), predicted distribution (must be > 0 where p > 0)

    Returns:
        cross-entropy in bits
    """
    raise NotImplementedError


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence KL(p || q) in bits.

    KL(p||q) = sum_i p_i * log2(p_i / q_i)
             = H(p, q) - H(p)

    Properties to verify:
    - KL(p||q) >= 0 always (Gibbs' inequality)
    - KL(p||q) = 0 iff p = q
    - KL is NOT symmetric: KL(p||q) != KL(q||p) in general

    Args:
        p: shape (K,), true distribution
        q: shape (K,), approximate distribution

    Returns:
        KL divergence in bits (non-negative)
    """
    raise NotImplementedError


def mutual_information(joint_p: np.ndarray) -> float:
    """Compute mutual information I(X; Y) from a joint distribution table.

    I(X;Y) = sum_{x,y} p(x,y) * log2(p(x,y) / (p(x) * p(y)))
           = KL(p(x,y) || p(x)p(y))

    I(X;Y) measures how much knowing X tells you about Y (and vice versa).
    I(X;Y) = 0 iff X and Y are independent.

    Args:
        joint_p: shape (n_x, n_y), joint probability table.
                 joint_p[i, j] = P(X=x_i, Y=y_j)

    Returns:
        mutual information in bits (non-negative)
    """
    raise NotImplementedError


def differential_entropy_gaussian(sigma: float) -> float:
    """Differential entropy of a univariate Gaussian N(mu, sigma^2).

    Derive analytically:
        h(X) = 0.5 * ln(2 * pi * e * sigma^2)

    Note: this is in nats (natural log), not bits.
    The Gaussian has the maximum entropy among all distributions with the
    same variance.

    Args:
        sigma: standard deviation

    Returns:
        differential entropy in nats
    """
    raise NotImplementedError


def kl_divergence_gaussians(
    mu1: float, sigma1: float, mu2: float, sigma2: float
) -> float:
    """KL divergence between two univariate Gaussians (in nats).

    Derive the closed form:
        KL(N(mu1,s1^2) || N(mu2,s2^2)) =
            log(s2/s1) + (s1^2 + (mu1-mu2)^2) / (2*s2^2) - 0.5

    Args:
        mu1, sigma1: parameters of p
        mu2, sigma2: parameters of q

    Returns:
        KL divergence in nats (non-negative)
    """
    raise NotImplementedError
