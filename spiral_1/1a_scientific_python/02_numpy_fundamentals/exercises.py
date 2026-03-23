"""Module 02: NumPy Fundamentals — Exercise Stubs

Implement each function below until all tests in tests/test_exercises.py pass.
All implementations should use NumPy vectorized operations (no Python for-loops over elements).
"""

import numpy as np
from numpy.typing import NDArray


def normalize_array(arr: NDArray[np.floating]) -> NDArray[np.floating]:
    """Normalize an array to have mean=0 and standard deviation=1 (z-score normalization).

    Args:
        arr: Input array of any shape with floating-point values.

    Returns:
        A new array of the same shape with mean approximately 0 and std approximately 1.
        If the standard deviation is 0 (constant array), return an array of zeros.
    """
    raise NotImplementedError


def euclidean_distance(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute pairwise Euclidean distances between two sets of points.

    Args:
        a: Array of shape (n, d) — n points in d dimensions.
        b: Array of shape (m, d) — m points in d dimensions.

    Returns:
        Array of shape (n, m) where element [i, j] is the Euclidean distance
        between a[i] and b[j].
    """
    raise NotImplementedError


def moving_average(arr: NDArray[np.floating], window: int) -> NDArray[np.floating]:
    """Compute the simple moving average of a 1D array.

    Args:
        arr: 1D input array.
        window: Size of the sliding window. Must be >= 1 and <= len(arr).

    Returns:
        1D array of length len(arr) - window + 1, where each element is the
        mean of the corresponding window of values from the input.
    """
    raise NotImplementedError


def one_hot_encode(labels: NDArray[np.integer], num_classes: int) -> NDArray[np.floating]:
    """Convert an array of integer labels to one-hot encoded vectors.

    Args:
        labels: 1D array of integer labels in range [0, num_classes).
        num_classes: Total number of classes.

    Returns:
        2D array of shape (len(labels), num_classes) with dtype float64.
        Each row has a 1.0 at the index corresponding to the label and 0.0 elsewhere.
    """
    raise NotImplementedError


def batch_dot_product(a: NDArray[np.floating], b: NDArray[np.floating]) -> NDArray[np.floating]:
    """Compute element-wise dot products for batches of vectors.

    Args:
        a: Array of shape (batch_size, d).
        b: Array of shape (batch_size, d).

    Returns:
        1D array of shape (batch_size,) where element [i] is the dot product
        of a[i] and b[i].
    """
    raise NotImplementedError
