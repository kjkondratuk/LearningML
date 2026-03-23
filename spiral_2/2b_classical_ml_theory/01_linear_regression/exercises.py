"""
Linear Regression Exercises -- Spiral 2, Phase 2B, Module 01

Derive each estimator on paper before implementing.
No scikit-learn. NumPy only.
"""

import numpy as np


def ols_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Ordinary Least Squares via the normal equations.

    Derive: minimize ||Xw - y||^2 => w = (X^T X)^{-1} X^T y.

    Explain when this breaks: X^T X is singular when features are
    linearly dependent or n < d.

    Args:
        X: shape (n, d), design matrix
        y: shape (n,), targets

    Returns:
        w: shape (d,), OLS weights
    """
    raise NotImplementedError


def ols_via_gradient_descent(
    X: np.ndarray, y: np.ndarray, lr: float, n_steps: int
) -> np.ndarray:
    """OLS via batch gradient descent on MSE loss.

    Derive: grad_w MSE = (2/n) X^T (Xw - y).

    Should converge to the same solution as closed form.

    Args:
        X: shape (n, d)
        y: shape (n,)
        lr: learning rate
        n_steps: number of gradient descent steps

    Returns:
        w: shape (d,), learned weights
    """
    raise NotImplementedError


def ols_via_svd(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS via SVD pseudoinverse.

    w = X^+ y where X^+ = V S^+ U^T.

    More numerically stable than the normal equations because it
    avoids forming X^T X (which squares the condition number).

    Args:
        X: shape (n, d)
        y: shape (n,)

    Returns:
        w: shape (d,)
    """
    raise NotImplementedError


def mle_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """MLE under Gaussian noise model y = Xw + epsilon, epsilon ~ N(0, sigma^2).

    Derive: the MLE for w is the same as OLS. The MLE for sigma^2 is
    the mean squared residual.

    Args:
        X: shape (n, d)
        y: shape (n,)

    Returns:
        w: shape (d,), MLE weights (same as OLS)
    """
    raise NotImplementedError


def map_linear_regression(
    X: np.ndarray, y: np.ndarray, prior_precision: float
) -> np.ndarray:
    """MAP estimate with Gaussian prior N(0, prior_precision^{-1} I) on weights.

    Derive: w_MAP = (X^T X + prior_precision * I)^{-1} X^T y.
    This IS ridge regression. The prior_precision is the regularization strength.

    Args:
        X: shape (n, d)
        y: shape (n,)
        prior_precision: tau -- larger means stronger regularization

    Returns:
        w: shape (d,), MAP weights
    """
    raise NotImplementedError


def bayesian_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    prior_precision: float,
    noise_precision: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Full Bayesian posterior over weights.

    Prior: w ~ N(0, prior_precision^{-1} I)
    Likelihood: y | X, w ~ N(Xw, noise_precision^{-1} I)

    Posterior: w | X, y ~ N(mu_post, Sigma_post) where:
        Sigma_post = (prior_precision * I + noise_precision * X^T X)^{-1}
        mu_post = noise_precision * Sigma_post @ X^T @ y

    Args:
        X: shape (n, d)
        y: shape (n,)
        prior_precision: precision of the prior (1/sigma_prior^2)
        noise_precision: precision of the noise (1/sigma_noise^2)

    Returns:
        mu_post: shape (d,), posterior mean
        Sigma_post: shape (d, d), posterior covariance
    """
    raise NotImplementedError


def predictive_distribution(
    X_test: np.ndarray,
    posterior_mean: np.ndarray,
    posterior_cov: np.ndarray,
    noise_precision: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Predictive distribution at test points.

    p(y* | x*, X, y) = N(y* | x*^T mu_post, 1/beta + x*^T Sigma_post x*)

    where beta = noise_precision.

    The predictive variance has two components:
    1. Noise variance (1/beta) -- irreducible
    2. Epistemic uncertainty (x*^T Sigma_post x*) -- grows far from training data

    Args:
        X_test: shape (m, d), test inputs
        posterior_mean: shape (d,)
        posterior_cov: shape (d, d)
        noise_precision: beta

    Returns:
        pred_mean: shape (m,), predicted means
        pred_var: shape (m,), predicted variances
    """
    raise NotImplementedError


def polynomial_regression(
    X: np.ndarray, y: np.ndarray, degree: int
) -> np.ndarray:
    """Polynomial regression via basis expansion + OLS.

    Expand x -> [1, x, x^2, ..., x^degree] then fit OLS.

    Shows overfitting at high degree when n is small.

    Args:
        X: shape (n, 1), input feature
        y: shape (n,), targets
        degree: polynomial degree

    Returns:
        w: shape (degree+1,), polynomial coefficients
    """
    raise NotImplementedError
