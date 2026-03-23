"""
Logistic Regression Exercises -- Spiral 2, Phase 2B, Module 02

Derive each function's math on paper before implementing.
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """The sigmoid (logistic) function.

    sigma(z) = 1 / (1 + exp(-z))

    Derive as the canonical link for Bernoulli: if log(p/(1-p)) = z, then p = sigma(z).
    Must be numerically stable for large positive and negative z.

    Args:
        z: any shape

    Returns:
        sigmoid values, same shape
    """
    raise NotImplementedError


def log_likelihood_binary(
    X: np.ndarray, y: np.ndarray, w: np.ndarray
) -> float:
    """Log-likelihood for binary logistic regression.

    LL = sum(y_i * log(sigma(x_i^T w)) + (1-y_i) * log(1-sigma(x_i^T w)))

    Must be numerically stable.

    Args:
        X: shape (n, d)
        y: shape (n,), binary labels in {0, 1}
        w: shape (d,), weights

    Returns:
        scalar log-likelihood
    """
    raise NotImplementedError


def gradient_log_likelihood(
    X: np.ndarray, y: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Gradient of log-likelihood.

    Derive: grad = X^T (y - sigma(Xw))

    Args:
        X: shape (n, d)
        y: shape (n,)
        w: shape (d,)

    Returns:
        gradient: shape (d,)
    """
    raise NotImplementedError


def hessian_log_likelihood(
    X: np.ndarray, y: np.ndarray, w: np.ndarray
) -> np.ndarray:
    """Hessian of log-likelihood.

    Derive: H = -X^T diag(s*(1-s)) X where s = sigma(Xw).
    Show H is negative semi-definite (log-likelihood is concave).

    Args:
        X: shape (n, d)
        y: shape (n,)
        w: shape (d,)

    Returns:
        H: shape (d, d), negative semi-definite
    """
    raise NotImplementedError


def logistic_regression_gd(
    X: np.ndarray, y: np.ndarray, lr: float, n_steps: int
) -> np.ndarray:
    """Fit binary logistic regression via gradient ascent on log-likelihood.

    Args:
        X: shape (n, d)
        y: shape (n,)
        lr: learning rate
        n_steps: iterations

    Returns:
        w: shape (d,), learned weights
    """
    raise NotImplementedError


def logistic_regression_newton(
    X: np.ndarray, y: np.ndarray, n_steps: int
) -> np.ndarray:
    """Fit via Newton-Raphson (IRLS).

    Derive: w_{new} = w - H^{-1} @ grad
    Show faster convergence than GD.

    Args:
        X: shape (n, d)
        y: shape (n,)
        n_steps: number of Newton steps

    Returns:
        w: shape (d,)
    """
    raise NotImplementedError


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax.

    softmax(z)_k = exp(z_k) / sum_j exp(z_j)

    Subtract max(z) for stability. Derive as generalization of sigmoid to K classes.

    Args:
        z: shape (n, K) or (K,)

    Returns:
        probabilities, same shape, rows sum to 1
    """
    raise NotImplementedError


def multinomial_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    lr: float,
    n_steps: int,
) -> np.ndarray:
    """Multi-class logistic regression via cross-entropy loss and GD.

    Args:
        X: shape (n, d)
        y: shape (n,), integer class labels 0..n_classes-1
        n_classes: number of classes
        lr: learning rate
        n_steps: iterations

    Returns:
        W: shape (d, n_classes), weight matrix
    """
    raise NotImplementedError


def decision_boundary(
    w: np.ndarray, b: float, x_range: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the linear decision boundary w^T x + b = 0 for 2D.

    Args:
        w: shape (2,), weight vector (excluding bias)
        b: bias scalar
        x_range: (min, max) for x1 axis

    Returns:
        x1: shape (100,), x1 values
        x2: shape (100,), corresponding x2 values on the boundary
    """
    raise NotImplementedError
