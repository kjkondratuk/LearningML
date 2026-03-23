"""
Module 04: Mini-Project — Compare Tree Depths (Overfitting U-Curve)

Train decision trees at various depths and record train/test accuracy
to visualize the bias-variance tradeoff.
"""

import numpy as np


def compare_tree_depths(
    X: np.ndarray,
    y: np.ndarray,
    depths: list = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Train decision trees at various max_depth values and record performance.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    depths : list of int or None values, max_depth settings to try.
        Defaults to [1, 2, 3, 5, 10, 20, None].
    test_size : float
    random_state : int

    Returns
    -------
    dict with keys:
        "depths"      : list, the depth values used
        "train_accs"  : list[float], training accuracy for each depth
        "test_accs"   : list[float], test accuracy for each depth
        "best_depth"  : int or None, depth with highest test accuracy
    """
    raise NotImplementedError
