"""
Mini-Project: Random Forest From Scratch
==========================================

Spiral 2, Phase 2B, Module 06

1. Implement full random forest (bagging + random features + decision trees)
2. Train on Iris and a synthetic dataset (100 features, 10 informative)
3. Compare against sklearn's RandomForestClassifier
4. Plot OOB error vs number of trees
5. Plot feature importance bar chart
"""

import numpy as np


def predict_forest(trees: list[dict], X: np.ndarray) -> np.ndarray:
    """Predict by majority vote across trees.

    Args:
        trees: list of tree dicts
        X: shape (n, d)

    Returns:
        predictions: shape (n,)
    """
    raise NotImplementedError


def oob_error(
    X: np.ndarray,
    y: np.ndarray,
    trees: list[dict],
    oob_indices: list[np.ndarray],
) -> float:
    """Compute out-of-bag error.

    For each sample, predict using only trees where it was NOT in the
    bootstrap sample.

    Args:
        X, y: full training data
        trees: list of tree dicts
        oob_indices: for each tree, array of indices NOT in its bootstrap sample

    Returns:
        OOB error rate
    """
    raise NotImplementedError


def compare_with_sklearn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    our_trees: list[dict],
) -> dict:
    """Compare our implementation against sklearn.

    Returns:
        dict with 'our_accuracy', 'sklearn_accuracy'
    """
    raise NotImplementedError
