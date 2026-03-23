"""
Module 04: Decision Trees & Ensembles — Exercise Stubs

Implement each function until `pytest tests/test_exercises.py` passes.
"""

import numpy as np


def gini_impurity(y: np.ndarray) -> float:
    """Compute Gini impurity for a set of labels.

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,), integer class labels

    Returns
    -------
    float in [0, 0.5] for binary, [0, 1-1/K] for K classes
    """
    raise NotImplementedError


def information_gain(
    y_parent: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
) -> float:
    """Compute information gain of a split using Gini impurity.

    Parameters
    ----------
    y_parent : np.ndarray, labels before split
    y_left   : np.ndarray, labels in the left child
    y_right  : np.ndarray, labels in the right child

    Returns
    -------
    float >= 0
    """
    raise NotImplementedError


def find_best_split(
    X: np.ndarray, y: np.ndarray
) -> dict:
    """Find the feature and threshold that maximize information gain.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)

    Returns
    -------
    dict with keys:
        "feature_index" : int
        "threshold"     : float
        "info_gain"     : float
    """
    raise NotImplementedError


def fit_decision_stump(
    X: np.ndarray, y: np.ndarray
) -> dict:
    """Fit a depth-1 decision tree (decision stump).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)

    Returns
    -------
    dict with keys:
        "feature_index" : int, which feature to split on
        "threshold"     : float, split threshold
        "left_label"    : int/float, predicted class for left child (feature <= threshold)
        "right_label"   : int/float, predicted class for right child (feature > threshold)
    """
    raise NotImplementedError


def sklearn_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 100,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """Train a random forest and return test predictions and train accuracy.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    y_train : np.ndarray of shape (n_train,)
    X_test  : np.ndarray of shape (n_test, n_features)
    n_estimators : int
    random_state : int

    Returns
    -------
    y_pred     : np.ndarray of shape (n_test,)
    train_acc  : float, accuracy on training set
    """
    raise NotImplementedError


def sklearn_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """Train gradient boosting and return test predictions and train accuracy.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    y_train : np.ndarray of shape (n_train,)
    X_test  : np.ndarray of shape (n_test, n_features)
    n_estimators : int
    learning_rate : float
    random_state : int

    Returns
    -------
    y_pred     : np.ndarray of shape (n_test,)
    train_acc  : float, accuracy on training set
    """
    raise NotImplementedError
