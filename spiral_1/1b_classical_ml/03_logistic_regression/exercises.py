"""
Module 03: Logistic Regression — Exercise Stubs

Implement each function until `pytest tests/test_exercises.py` passes.
"""

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function element-wise.

    Parameters
    ----------
    z : np.ndarray of any shape

    Returns
    -------
    np.ndarray of same shape, values in (0, 1)
    """
    raise NotImplementedError


def predict_proba(X: np.ndarray, w: np.ndarray, b: float) -> np.ndarray:
    """Compute predicted probabilities for binary classification.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    w : np.ndarray of shape (n_features,)
    b : float

    Returns
    -------
    np.ndarray of shape (n_samples,), values in (0, 1)
    """
    raise NotImplementedError


def predict_class(
    X: np.ndarray, w: np.ndarray, b: float, threshold: float = 0.5
) -> np.ndarray:
    """Predict binary class labels.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    w : np.ndarray of shape (n_features,)
    b : float
    threshold : float, classification threshold

    Returns
    -------
    np.ndarray of shape (n_samples,), values in {0, 1}
    """
    raise NotImplementedError


def compute_log_loss(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Compute binary cross-entropy (log) loss.

    Parameters
    ----------
    y_true  : np.ndarray of shape (n_samples,), values in {0, 1}
    y_proba : np.ndarray of shape (n_samples,), predicted probabilities in (0, 1)

    Returns
    -------
    float
    """
    raise NotImplementedError


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.1,
    n_iterations: int = 1000,
) -> tuple[np.ndarray, float, list[float]]:
    """Train logistic regression using gradient descent on log loss.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,), binary labels {0, 1}
    learning_rate : float
    n_iterations : int

    Returns
    -------
    w : np.ndarray of shape (n_features,)
    b : float
    losses : list[float], log loss at each iteration
    """
    raise NotImplementedError


def sklearn_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit logistic regression with sklearn and return predictions.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    y_train : np.ndarray of shape (n_train,)
    X_test  : np.ndarray of shape (n_test, n_features)

    Returns
    -------
    y_pred  : np.ndarray of shape (n_test,), predicted class labels
    y_proba : np.ndarray of shape (n_test,), predicted probability of class 1
    """
    raise NotImplementedError
