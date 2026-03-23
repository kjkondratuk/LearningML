"""Module 02: Mini-Project — K-Nearest Neighbors Classifier

Implement KNN using only NumPy. No scikit-learn allowed.
Run tests with: pytest tests/test_mini_project.py -v
"""

import numpy as np
from numpy.typing import NDArray


def knn_classify(
    train_features: NDArray[np.floating],
    train_labels: NDArray[np.integer],
    test_features: NDArray[np.floating],
    k: int = 3,
) -> NDArray[np.integer]:
    """Classify test points using K-Nearest Neighbors.

    For each test point:
    1. Compute the Euclidean distance to every training point.
    2. Find the k training points with the smallest distances.
    3. Assign the most common label among those k neighbors (majority vote).
       Break ties by choosing the label with the nearest single neighbor.

    Args:
        train_features: Array of shape (n_train, d) — training feature vectors.
        train_labels: Array of shape (n_train,) — integer class labels for training data.
        test_features: Array of shape (n_test, d) — test feature vectors to classify.
        k: Number of neighbors to consider.

    Returns:
        Array of shape (n_test,) — predicted integer class labels for each test point.
    """
    raise NotImplementedError
