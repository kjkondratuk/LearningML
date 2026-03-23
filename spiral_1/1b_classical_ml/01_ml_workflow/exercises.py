"""
Module 01: ML Workflow — Exercise Stubs

Implement each function until `pytest tests/test_exercises.py` passes.
"""

import numpy as np


def train_test_split_manual(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split feature matrix X and target vector y into train and test sets.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    test_size : float, fraction of data to reserve for testing (0 < test_size < 1)
    random_state : int, seed for reproducible shuffling

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    raise NotImplementedError


def compute_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute classification accuracy.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
    y_pred : np.ndarray of shape (n_samples,)

    Returns
    -------
    float in [0, 1]
    """
    raise NotImplementedError


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
    y_pred : np.ndarray of shape (n_samples,)

    Returns
    -------
    float >= 0
    """
    raise NotImplementedError


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean absolute error.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,)
    y_pred : np.ndarray of shape (n_samples,)

    Returns
    -------
    float >= 0
    """
    raise NotImplementedError


def detect_data_leakage(
    X_train: np.ndarray, X_test: np.ndarray
) -> dict:
    """Detect whether any rows in X_test also appear in X_train.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    X_test  : np.ndarray of shape (n_test, n_features)

    Returns
    -------
    dict with keys:
        "has_leakage" : bool
        "leaked_count" : int, number of test rows found in training set
    """
    raise NotImplementedError
