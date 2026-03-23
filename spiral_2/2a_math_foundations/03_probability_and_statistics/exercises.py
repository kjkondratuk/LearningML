"""
Probability and Statistics Exercises -- Spiral 2, Phase 2A, Module 03

Implement all distributions and estimators from the formulas.
scipy.stats is used ONLY in tests as a reference oracle.
"""

import numpy as np


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Compute the Gaussian (normal) PDF.

    p(x) = (1 / sqrt(2 pi sigma^2)) * exp(-(x - mu)^2 / (2 sigma^2))

    Verify against scipy.stats.norm.pdf in tests.

    Args:
        x: points at which to evaluate, any shape
        mu: mean
        sigma: standard deviation (not variance)

    Returns:
        pdf values, same shape as x
    """
    raise NotImplementedError


def multivariate_gaussian_pdf(
    x: np.ndarray, mu: np.ndarray, cov: np.ndarray
) -> float:
    """Compute the multivariate Gaussian PDF at a single point.

    p(x) = (2pi)^{-d/2} |cov|^{-1/2} exp(-0.5 (x-mu)^T cov^{-1} (x-mu))

    Requires matrix inverse and determinant -- use your Phase 2A Module 01
    implementations or np.linalg for now.

    Args:
        x: shape (d,)
        mu: shape (d,)
        cov: shape (d, d), positive definite covariance matrix

    Returns:
        scalar pdf value
    """
    raise NotImplementedError


def mle_gaussian(data: np.ndarray) -> tuple[float, float]:
    """Maximum Likelihood Estimation for a univariate Gaussian.

    Derive:
        mu_MLE = (1/N) sum(x_i)           -- sample mean
        sigma2_MLE = (1/N) sum((x_i - mu)^2) -- biased sample variance

    Note: this is the BIASED variance (divides by N, not N-1).

    Args:
        data: shape (N,), observed samples

    Returns:
        (mu_mle, sigma2_mle): mean and variance
    """
    raise NotImplementedError


def map_gaussian(
    data: np.ndarray,
    prior_mu: float,
    prior_sigma: float,
    likelihood_sigma: float,
) -> float:
    """MAP estimate for the mean of a Gaussian with a Gaussian prior.

    Derive: the MAP estimate is a precision-weighted combination:
        mu_MAP = (prior_precision * prior_mu + N * likelihood_precision * x_bar)
                 / (prior_precision + N * likelihood_precision)

    where precision = 1/sigma^2.

    Show that as N -> infinity, mu_MAP -> x_bar (data overwhelms the prior).
    Show that as prior_sigma -> 0, mu_MAP -> prior_mu (prior dominates).

    Args:
        data: shape (N,), observed samples
        prior_mu: prior mean on mu
        prior_sigma: prior standard deviation on mu
        likelihood_sigma: known standard deviation of the data

    Returns:
        mu_map: the MAP estimate for mu
    """
    raise NotImplementedError


def bayesian_update(
    prior: np.ndarray,
    likelihood: np.ndarray,
    evidence_grid: np.ndarray,
) -> np.ndarray:
    """Implement Bayes' rule on a discrete grid.

    posterior(theta) = prior(theta) * likelihood(theta) / Z
    where Z = sum(prior * likelihood) is the normalizing constant.

    Args:
        prior: shape (K,), prior probabilities over a grid of theta values
        likelihood: shape (K,), likelihood of observed data for each theta
        evidence_grid: shape (K,), the theta values (for documentation)

    Returns:
        posterior: shape (K,), normalized posterior probabilities
    """
    raise NotImplementedError


def conjugate_prior_beta_binomial(
    alpha: float, beta: float, n_successes: int, n_trials: int
) -> tuple[float, float]:
    """Compute the posterior Beta distribution for a Beta-Binomial model.

    Prior: p ~ Beta(alpha, beta)
    Likelihood: k successes in n trials ~ Binomial(n, p)
    Posterior: p ~ Beta(alpha + k, beta + n - k)

    No integrals needed -- conjugacy gives us the answer directly.

    Args:
        alpha: prior alpha parameter
        beta: prior beta parameter
        n_successes: observed successes
        n_trials: total trials

    Returns:
        (alpha_post, beta_post): posterior Beta parameters
    """
    raise NotImplementedError


def sample_from_distribution(
    pdf: callable,
    x_range: tuple[float, float],
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    """Sample from an arbitrary distribution using rejection sampling.

    Algorithm:
    1. Find M = max of pdf over x_range (approximate via grid)
    2. Repeat until n_samples accepted:
       a. Draw x ~ Uniform(x_range)
       b. Draw u ~ Uniform(0, M)
       c. If u < pdf(x), accept x

    Args:
        pdf: probability density function (callable)
        x_range: (min, max) domain
        n_samples: number of samples to generate
        seed: random seed

    Returns:
        samples: shape (n_samples,)
    """
    raise NotImplementedError


def central_limit_theorem_demo(
    distribution: callable,
    sample_sizes: list[int],
    n_experiments: int = 1000,
    seed: int = 42,
) -> dict[int, np.ndarray]:
    """Demonstrate the Central Limit Theorem.

    For each sample_size N:
    1. Draw n_experiments batches of N samples from the distribution
    2. Compute the sample mean for each batch
    3. Collect all sample means

    As N grows, the distribution of sample means approaches a Gaussian,
    regardless of the original distribution.

    Args:
        distribution: callable(size, rng) that draws samples
        sample_sizes: list of N values to test
        n_experiments: how many batches per sample size
        seed: random seed

    Returns:
        dict mapping sample_size -> array of n_experiments sample means
    """
    raise NotImplementedError
