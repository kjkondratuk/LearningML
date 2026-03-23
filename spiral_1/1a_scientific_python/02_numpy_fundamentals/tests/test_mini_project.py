"""Tests for Module 02: Mini-Project — KNN Classifier.

Run with: pytest tests/test_mini_project.py -v
"""

import numpy as np
import pytest

from mini_project import knn_classify


def _make_blobs(n_per_class: int, n_classes: int, d: int, seed: int = 42):
    """Generate well-separated clusters for testing."""
    rng = np.random.RandomState(seed)
    features_list = []
    labels_list = []
    for c in range(n_classes):
        center = rng.randn(d) * 10
        points = center + rng.randn(n_per_class, d) * 0.5
        features_list.append(points)
        labels_list.append(np.full(n_per_class, c, dtype=np.intp))
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    return features, labels


class TestKNNClassify:
    def test_perfect_classification_on_well_separated_blobs(self):
        train_x, train_y = _make_blobs(50, 3, 5, seed=0)
        test_x, test_y = _make_blobs(10, 3, 5, seed=1)
        preds = knn_classify(train_x, train_y, test_x, k=3)
        accuracy = (preds == test_y).mean()
        assert accuracy >= 0.9, f"Expected >= 90% accuracy, got {accuracy:.1%}"

    def test_k_equals_one(self):
        """With k=1, the nearest training point's label should be returned."""
        train_x = np.array([[0.0, 0.0], [10.0, 10.0]])
        train_y = np.array([0, 1])
        test_x = np.array([[0.1, 0.1], [9.9, 9.9]])
        preds = knn_classify(train_x, train_y, test_x, k=1)
        np.testing.assert_array_equal(preds, [0, 1])

    def test_output_shape(self):
        train_x, train_y = _make_blobs(20, 2, 4)
        test_x = np.random.randn(15, 4)
        preds = knn_classify(train_x, train_y, test_x, k=3)
        assert preds.shape == (15,)

    def test_output_dtype_is_integer(self):
        train_x, train_y = _make_blobs(20, 2, 3)
        test_x = np.random.randn(5, 3)
        preds = knn_classify(train_x, train_y, test_x, k=3)
        assert np.issubdtype(preds.dtype, np.integer)

    def test_two_classes_simple(self):
        """Sanity check: points near class 0's center should be classified as 0."""
        train_x = np.array([
            [0.0, 0.0], [0.1, 0.1], [0.2, 0.0],  # class 0 cluster
            [5.0, 5.0], [5.1, 5.1], [5.2, 5.0],  # class 1 cluster
        ])
        train_y = np.array([0, 0, 0, 1, 1, 1])
        test_x = np.array([[0.05, 0.05], [5.05, 5.05]])
        preds = knn_classify(train_x, train_y, test_x, k=3)
        np.testing.assert_array_equal(preds, [0, 1])

    def test_larger_k(self):
        train_x, train_y = _make_blobs(100, 4, 8, seed=77)
        test_x, test_y = _make_blobs(20, 4, 8, seed=88)
        preds = knn_classify(train_x, train_y, test_x, k=7)
        accuracy = (preds == test_y).mean()
        assert accuracy >= 0.85, f"Expected >= 85% accuracy with k=7, got {accuracy:.1%}"
