"""
Module 06: Unsupervised Learning — Exercise Stubs

Implement each function until `pytest tests/test_exercises.py` passes.
"""

import numpy as np


def kmeans_assign(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each data point to the nearest centroid.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    centroids : np.ndarray of shape (k, n_features)

    Returns
    -------
    np.ndarray of shape (n_samples,), integer cluster labels (0 to k-1)
    """
    raise NotImplementedError


def kmeans_update(X: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    """Recompute centroids as the mean of assigned points.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    labels : np.ndarray of shape (n_samples,), cluster assignments
    k : int, number of clusters

    Returns
    -------
    np.ndarray of shape (k, n_features), updated centroids
    """
    raise NotImplementedError


def kmeans_from_scratch(
    X: np.ndarray,
    k: int,
    max_iter: int = 100,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run K-means clustering from scratch.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    k : int, number of clusters
    max_iter : int, maximum iterations
    random_state : int

    Returns
    -------
    labels    : np.ndarray of shape (n_samples,), final cluster assignments
    centroids : np.ndarray of shape (k, n_features), final centroids
    n_iters   : int, number of iterations until convergence
    """
    raise NotImplementedError


def find_optimal_k(
    X: np.ndarray,
    max_k: int = 10,
    random_state: int = 42,
) -> list[float]:
    """Compute inertia (sum of squared distances to nearest centroid) for K=1..max_k.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    max_k : int
    random_state : int

    Returns
    -------
    list[float] of length max_k, where element i is the inertia for K=i+1
    """
    raise NotImplementedError


def sklearn_pca(
    X: np.ndarray, n_components: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce dimensionality using sklearn PCA.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    n_components : int

    Returns
    -------
    X_reduced         : np.ndarray of shape (n_samples, n_components)
    explained_variance : np.ndarray of shape (n_components,), fraction of variance explained
    """
    raise NotImplementedError


def sklearn_hierarchical(
    X: np.ndarray, n_clusters: int = 3
) -> np.ndarray:
    """Cluster using sklearn AgglomerativeClustering.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    n_clusters : int

    Returns
    -------
    np.ndarray of shape (n_samples,), cluster labels
    """
    raise NotImplementedError
