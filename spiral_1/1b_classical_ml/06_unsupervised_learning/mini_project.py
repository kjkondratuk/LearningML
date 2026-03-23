"""
Module 06: Mini-Project — Customer Segmentation

Combine standardization, K-means, and PCA into a customer segmentation pipeline.
"""

import numpy as np


def segment_customers(
    X: np.ndarray,
    max_k: int = 10,
    random_state: int = 42,
) -> dict:
    """Segment customers using K-means with PCA visualization.

    Steps:
    1. Standardize features (zero mean, unit variance).
    2. Find optimal K using the elbow method (inertias for K=1..max_k).
    3. Choose K at the elbow (largest drop in inertia).
    4. Run K-means with chosen K.
    5. Reduce to 2D with PCA for visualization.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    max_k : int
    random_state : int

    Returns
    -------
    dict with keys:
        "labels"           : np.ndarray of shape (n_samples,), cluster assignments
        "centroids"        : np.ndarray of shape (k, n_features), in standardized space
        "chosen_k"         : int, the selected number of clusters
        "inertias"         : list[float], inertia for K=1..max_k
        "pca_coords"       : np.ndarray of shape (n_samples, 2), 2D projection
        "explained_variance" : np.ndarray of shape (2,), variance explained by each PC
    """
    raise NotImplementedError
