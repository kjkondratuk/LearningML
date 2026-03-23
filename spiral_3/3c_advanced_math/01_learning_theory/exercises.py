"""
Statistical Learning Theory Exercises

Implement bounds and complexity measures from:
    Shalev-Shwartz & Ben-David, "Understanding Machine Learning" ch. 3-6
"""

import numpy as np
import torch


def vc_dimension_linear_classifier(d: int) -> int:
    """VC dimension of linear classifiers in R^d.

    VC(H) = d + 1 for the class of hyperplanes in R^d.

    Args:
        d: Input dimensionality.

    Returns:
        VC dimension.
    """
    raise NotImplementedError


def pac_sample_complexity(epsilon: float, delta: float, vc_dim: int) -> int:
    """Compute PAC sample complexity bound.

    m >= (1/epsilon) * (vc_dim * log(1/epsilon) + log(1/delta))

    Args:
        epsilon: Accuracy parameter (maximum excess risk).
        delta: Confidence parameter (failure probability).
        vc_dim: VC dimension of the hypothesis class.

    Returns:
        Minimum number of samples needed.
    """
    raise NotImplementedError


def rademacher_complexity_linear(
    X: np.ndarray,
    num_samples: int = 1000,
) -> float:
    """Empirical Rademacher complexity for linear classifiers.

    R_n(H) = E_sigma[ sup_w (1/n) sum_i sigma_i * w^T x_i ]

    For unit-norm w, this equals (1/n) * E_sigma[||sum sigma_i x_i||].

    Args:
        X: Data matrix, shape (n, d).
        num_samples: Number of Rademacher samples for Monte Carlo estimate.

    Returns:
        Empirical Rademacher complexity estimate.
    """
    raise NotImplementedError


def hoeffding_bound(n: int, t: float) -> float:
    """Hoeffding's inequality for bounded random variables in [0, 1].

    P(|mean(X_i) - E[X]| >= t) <= 2 * exp(-2 * n * t^2)

    Args:
        n: Number of samples.
        t: Deviation threshold.

    Returns:
        Upper bound on the probability.
    """
    raise NotImplementedError


def generalization_bound(
    train_error: float,
    n: int,
    vc_dim: int,
    delta: float,
) -> float:
    """VC generalization bound.

    test_error <= train_error + sqrt((vc_dim * (log(2n/vc_dim) + 1) + log(4/delta)) / n)

    Args:
        train_error: Empirical risk on training data.
        n: Number of training samples.
        vc_dim: VC dimension.
        delta: Confidence parameter.

    Returns:
        Upper bound on true risk.
    """
    raise NotImplementedError


def mcdiarmid_bound(
    f_values: np.ndarray,
    perturbation_bound: float,
    n: int,
    t: float,
) -> float:
    """McDiarmid's inequality.

    If changing any single input x_i changes f by at most c, then:
    P(f(X) - E[f(X)] >= t) <= exp(-2t^2 / (n * c^2))

    Args:
        f_values: Observed function values (for reference).
        perturbation_bound: Maximum change c when perturbing one input.
        n: Number of inputs.
        t: Deviation threshold.

    Returns:
        Upper bound on the tail probability.
    """
    raise NotImplementedError
