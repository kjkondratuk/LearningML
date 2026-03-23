"""
Module 07: Model Evaluation — Exercise Stubs

Implement each function until `pytest tests/test_exercises.py` passes.
"""

import numpy as np


def confusion_matrix_manual(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Build a 2x2 confusion matrix for binary classification.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,), values in {0, 1}
    y_pred : np.ndarray of shape (n_samples,), values in {0, 1}

    Returns
    -------
    np.ndarray of shape (2, 2) with layout:
        [[TN, FP],
         [FN, TP]]
    """
    raise NotImplementedError


def precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict[str, float]:
    """Compute precision, recall, and F1 score for binary classification.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,), values in {0, 1}
    y_pred : np.ndarray of shape (n_samples,), values in {0, 1}

    Returns
    -------
    dict with keys "precision", "recall", "f1", each a float
    """
    raise NotImplementedError


def roc_points(
    y_true: np.ndarray, y_scores: np.ndarray, n_thresholds: int = 100
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ROC curve points (FPR, TPR) at evenly spaced thresholds.

    Parameters
    ----------
    y_true : np.ndarray of shape (n_samples,), values in {0, 1}
    y_scores : np.ndarray of shape (n_samples,), predicted probabilities
    n_thresholds : int, number of thresholds to evaluate

    Returns
    -------
    fpr : np.ndarray of shape (n_thresholds,), false positive rates (sorted ascending)
    tpr : np.ndarray of shape (n_thresholds,), true positive rates (corresponding)
    """
    raise NotImplementedError


def cross_validate_manual(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 5,
    random_state: int = 42,
) -> list[float]:
    """Perform K-fold cross-validation using a simple classifier.

    Use sklearn's LogisticRegression as the classifier. Split the data into
    K folds yourself (do not use sklearn's cross_val_score or KFold).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    k : int, number of folds
    random_state : int, seed for shuffling before splitting

    Returns
    -------
    list[float] of length k, accuracy on each held-out fold
    """
    raise NotImplementedError


def learning_curve_data(
    X: np.ndarray,
    y: np.ndarray,
    train_sizes: list[float] = None,
    random_state: int = 42,
) -> dict:
    """Compute training and test scores at increasing training set sizes.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    train_sizes : list[float], fractions of training data to use.
        Defaults to [0.1, 0.2, 0.4, 0.6, 0.8, 1.0].
    random_state : int

    Returns
    -------
    dict with keys:
        "train_sizes"  : list[int], actual number of training samples used
        "train_scores" : list[float], training accuracy at each size
        "test_scores"  : list[float], test accuracy at each size
    """
    raise NotImplementedError
