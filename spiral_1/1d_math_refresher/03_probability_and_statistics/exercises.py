"""
Module 03: Probability & Statistics
=====================================
The probabilistic foundations of machine learning.
"""

import numpy as np


def estimate_probability(outcomes: np.ndarray, event_value: int) -> float:
    """Estimate the probability of an event from observed outcomes.

    P(event) = count(outcomes == event_value) / len(outcomes)

    Args:
        outcomes: 1D array of observed outcomes (integers).
        event_value: The value whose probability to estimate.

    Returns:
        Estimated probability (float in [0, 1]).
    """
    raise NotImplementedError


def bayes_theorem(
    p_a: float, p_b_given_a: float, p_b: float
) -> float:
    """Apply Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B).

    Args:
        p_a: Prior probability P(A).
        p_b_given_a: Likelihood P(B|A).
        p_b: Evidence P(B).

    Returns:
        Posterior probability P(A|B).
    """
    raise NotImplementedError


def bayes_disease_test(
    prevalence: float, sensitivity: float, specificity: float
) -> float:
    """Compute P(disease | positive test) using Bayes' theorem.

    Args:
        prevalence: P(disease) -- fraction of population with disease.
        sensitivity: P(positive | disease) -- true positive rate.
        specificity: P(negative | no disease) -- true negative rate.

    Returns:
        P(disease | positive test).

    Hint:
        P(positive) = P(positive|disease)*P(disease)
                     + P(positive|no disease)*P(no disease)
    """
    raise NotImplementedError


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Compute the Gaussian (normal) probability density function.

    f(x) = (1 / sqrt(2 * pi * sigma^2)) * exp(-(x - mu)^2 / (2 * sigma^2))

    Args:
        x: Points at which to evaluate the PDF.
        mu: Mean of the distribution.
        sigma: Standard deviation (> 0).

    Returns:
        PDF values, same shape as x.
    """
    raise NotImplementedError


def sample_and_estimate(
    mu: float, sigma: float, n_samples: int, seed: int = 42
) -> dict[str, float]:
    """Sample from a Gaussian and estimate its parameters.

    Args:
        mu: True mean.
        sigma: True standard deviation.
        n_samples: Number of samples to draw.
        seed: Random seed.

    Returns:
        Dict with keys:
        - "estimated_mean": sample mean
        - "estimated_std": sample standard deviation (ddof=1)
        - "mean_error": |estimated_mean - mu|
        - "std_error": |estimated_std - sigma|
    """
    raise NotImplementedError


def naive_bayes_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Gaussian Naive Bayes classifier from scratch.

    Assumes features are conditionally independent and Gaussian-distributed
    given the class label.

    Steps:
        1. For each class, compute the mean and std of each feature.
        2. For each test sample, compute log P(class) + sum(log P(feature|class))
           for each class.
        3. Predict the class with the highest log-probability.

    Args:
        X_train: Training features, shape (n_train, n_features).
        y_train: Training labels, shape (n_train,), integer class labels.
        X_test: Test features, shape (n_test, n_features).

    Returns:
        Predicted labels for X_test, shape (n_test,).
    """
    raise NotImplementedError
