"""
Decision Trees and Ensembles Exercises -- Spiral 2, Phase 2B, Module 06
"""

import numpy as np
from typing import Optional, Union


def gini_impurity(y: np.ndarray) -> float:
    """Compute Gini impurity.

    Gini = 1 - sum_k (p_k)^2 where p_k is the proportion of class k.
    Derive from misclassification probability.

    Args:
        y: shape (n,), class labels

    Returns:
        Gini impurity (0 = pure, max at uniform)
    """
    raise NotImplementedError


def entropy_impurity(y: np.ndarray) -> float:
    """Compute entropy-based impurity (in bits).

    H = -sum_k p_k log2(p_k)

    Args:
        y: shape (n,), class labels

    Returns:
        entropy in bits
    """
    raise NotImplementedError


def information_gain(y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Compute information gain for a split.

    IG = H(parent) - (n_left/n * H(left) + n_right/n * H(right))

    Args:
        y: parent labels
        y_left: left child labels
        y_right: right child labels

    Returns:
        information gain
    """
    raise NotImplementedError


def find_best_split(
    X: np.ndarray, y: np.ndarray, feature_idx: int
) -> tuple[float, float]:
    """Find the best split threshold for a single feature.

    Try all unique values as thresholds, pick the one with max IG.

    Args:
        X: shape (n, d)
        y: shape (n,)
        feature_idx: which feature to split on

    Returns:
        (threshold, info_gain): best threshold and its IG
    """
    raise NotImplementedError


def build_decision_tree(
    X: np.ndarray,
    y: np.ndarray,
    max_depth: int = None,
    min_samples: int = 2,
    max_features: Optional[int] = None,
    depth: int = 0,
) -> dict:
    """Recursively build a decision tree.

    Returns a tree structure as nested dicts:
    - Leaf: {'leaf': True, 'value': class_label, 'n_samples': n}
    - Internal: {'feature': idx, 'threshold': val, 'left': subtree, 'right': subtree}

    Args:
        X: shape (n, d)
        y: shape (n,)
        max_depth: maximum tree depth (None = unlimited)
        min_samples: minimum samples to split
        max_features: random subset of features to consider (None = all)
        depth: current depth (internal use)

    Returns:
        tree: nested dict
    """
    raise NotImplementedError


def predict_tree(tree: dict, X: np.ndarray) -> np.ndarray:
    """Predict class labels using a decision tree.

    Args:
        tree: tree structure from build_decision_tree
        X: shape (n, d)

    Returns:
        predictions: shape (n,)
    """
    raise NotImplementedError


def prune_tree(
    tree: dict, X_val: np.ndarray, y_val: np.ndarray
) -> dict:
    """Cost-complexity pruning (reduced error pruning).

    For each internal node, check if replacing it with a leaf improves
    validation accuracy. If so, prune.

    Args:
        tree: unpruned tree
        X_val, y_val: validation data

    Returns:
        pruned tree
    """
    raise NotImplementedError


def bagging_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    n_trees: int,
    max_depth: int = None,
    max_features: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """Bagging with random feature subsets (= Random Forest).

    For each tree:
    1. Bootstrap sample from (X, y)
    2. Build tree with random feature subsets at each split

    Args:
        X: shape (n, d)
        y: shape (n,)
        n_trees: number of trees
        max_depth: max depth per tree
        max_features: features to consider per split (default sqrt(d))
        seed: random seed

    Returns:
        list of tree dicts
    """
    raise NotImplementedError


def gradient_boosting(
    X: np.ndarray,
    y: np.ndarray,
    n_rounds: int,
    lr: float = 0.1,
    max_depth: int = 3,
    seed: int = 42,
) -> tuple[list[dict], float]:
    """Gradient boosting for regression.

    Each tree fits the residuals: r_t = y - F_{t-1}(x).
    F_t(x) = F_{t-1}(x) + lr * h_t(x).
    Derive: this is gradient descent in function space.

    Args:
        X: shape (n, d)
        y: shape (n,), continuous targets
        n_rounds: number of boosting rounds
        lr: learning rate (shrinkage)
        max_depth: depth of each weak learner
        seed: random seed

    Returns:
        trees: list of regression tree dicts
        initial_pred: the initial prediction (mean of y)
    """
    raise NotImplementedError


def feature_importance(trees: list[dict], n_features: int) -> np.ndarray:
    """Compute feature importance from an ensemble.

    Importance of feature j = total reduction in impurity from splits on j,
    averaged across all trees.

    Args:
        trees: list of tree dicts
        n_features: total number of features

    Returns:
        importances: shape (n_features,), normalized to sum to 1
    """
    raise NotImplementedError
