"""
Module 02: Linear Regression — Exercise Stubs

Implement each function until `pytest tests/test_exercises.py` passes.
"""

import numpy as np


def predict_linear(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Compute linear predictions: y_hat = X @ w + b.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    w : np.ndarray of shape (n_features,)
    b : float

    Returns
    -------
    np.ndarray of shape (n_samples,)
    """
    raise NotImplementedError


def compute_loss_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error loss.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
    y_pred : np.ndarray of shape (n_samples,)

    Returns
    -------
    float
    """
    raise NotImplementedError


def gradient_descent_step(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    b: float,
    learning_rate: float,
) -> tuple[np.ndarray, float]:
    """Perform one gradient descent update for linear regression with MSE loss.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    w : np.ndarray of shape (n_features,)
    b : float
    learning_rate : float

    Returns
    -------
    w_new : np.ndarray of shape (n_features,)
    b_new : float
    """
    raise NotImplementedError


def train_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.01,
    n_iterations: int = 1000,
) -> tuple[np.ndarray, float, list[float]]:
    """Train a linear regression model using gradient descent.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    learning_rate : float
    n_iterations : int

    Returns
    -------
    w : np.ndarray of shape (n_features,), learned weights
    b : float, learned bias
    losses : list[float], MSE loss at each iteration
    """
    raise NotImplementedError


def sklearn_linear_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a linear regression using sklearn and return predictions.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    y_train : np.ndarray of shape (n_train,)
    X_test  : np.ndarray of shape (n_test, n_features)

    Returns
    -------
    y_pred : np.ndarray of shape (n_test,), predictions on test set
    coef   : np.ndarray of shape (n_features,), learned coefficients
    intercept : float, learned intercept
    """
    raise NotImplementedError
