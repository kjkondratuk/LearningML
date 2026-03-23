"""Tests for Module 02: NumPy Fundamentals.

Run with: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    batch_dot_product,
    euclidean_distance,
    moving_average,
    normalize_array,
    one_hot_encode,
)


# ---------------------------------------------------------------------------
# normalize_array
# ---------------------------------------------------------------------------

class TestNormalizeArray:
    def test_mean_near_zero(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_array(arr)
        assert abs(result.mean()) < 1e-10

    def test_std_near_one(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_array(arr)
        assert abs(result.std() - 1.0) < 1e-10

    def test_2d_array(self):
        arr = np.random.randn(100, 50) * 5 + 10
        result = normalize_array(arr)
        assert abs(result.mean()) < 0.01
        assert abs(result.std() - 1.0) < 0.01

    def test_constant_array_returns_zeros(self):
        arr = np.full(10, 42.0)
        result = normalize_array(arr)
        np.testing.assert_array_equal(result, np.zeros(10))

    def test_preserves_shape(self):
        arr = np.random.randn(3, 4, 5)
        result = normalize_array(arr)
        assert result.shape == arr.shape

    def test_does_not_modify_input(self):
        arr = np.array([1.0, 2.0, 3.0])
        original = arr.copy()
        normalize_array(arr)
        np.testing.assert_array_equal(arr, original)


# ---------------------------------------------------------------------------
# euclidean_distance
# ---------------------------------------------------------------------------

class TestEuclideanDistance:
    def test_identical_points(self):
        a = np.array([[1.0, 2.0]])
        b = np.array([[1.0, 2.0]])
        result = euclidean_distance(a, b)
        np.testing.assert_allclose(result, [[0.0]], atol=1e-10)

    def test_known_distance(self):
        a = np.array([[0.0, 0.0]])
        b = np.array([[3.0, 4.0]])
        result = euclidean_distance(a, b)
        np.testing.assert_allclose(result, [[5.0]], atol=1e-10)

    def test_pairwise_shape(self):
        a = np.random.randn(5, 3)
        b = np.random.randn(7, 3)
        result = euclidean_distance(a, b)
        assert result.shape == (5, 7)

    def test_symmetry(self):
        a = np.random.randn(4, 3)
        b = np.random.randn(6, 3)
        d_ab = euclidean_distance(a, b)
        d_ba = euclidean_distance(b, a)
        np.testing.assert_allclose(d_ab, d_ba.T, atol=1e-10)

    def test_1d_points(self):
        a = np.array([[1.0], [2.0], [3.0]])
        b = np.array([[0.0], [5.0]])
        expected = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0]])
        np.testing.assert_allclose(euclidean_distance(a, b), expected, atol=1e-10)

    def test_all_distances_non_negative(self):
        a = np.random.randn(10, 5)
        b = np.random.randn(8, 5)
        result = euclidean_distance(a, b)
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# moving_average
# ---------------------------------------------------------------------------

class TestMovingAverage:
    def test_window_of_one(self):
        arr = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(moving_average(arr, 1), [1.0, 2.0, 3.0])

    def test_full_window(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0])
        result = moving_average(arr, 4)
        np.testing.assert_allclose(result, [2.5])

    def test_known_values(self):
        arr = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        result = moving_average(arr, 3)
        np.testing.assert_allclose(result, [3.0, 5.0, 7.0])

    def test_output_length(self):
        arr = np.random.randn(100)
        result = moving_average(arr, 10)
        assert len(result) == 91

    def test_constant_array(self):
        arr = np.full(20, 5.0)
        result = moving_average(arr, 5)
        np.testing.assert_allclose(result, np.full(16, 5.0))


# ---------------------------------------------------------------------------
# one_hot_encode
# ---------------------------------------------------------------------------

class TestOneHotEncode:
    def test_basic(self):
        labels = np.array([0, 1, 2])
        result = one_hot_encode(labels, 3)
        expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_repeated_labels(self):
        labels = np.array([1, 1, 0, 2, 0])
        result = one_hot_encode(labels, 3)
        assert result.shape == (5, 3)
        np.testing.assert_array_equal(result[0], [0, 1, 0])
        np.testing.assert_array_equal(result[3], [0, 0, 1])

    def test_output_dtype(self):
        labels = np.array([0, 1])
        result = one_hot_encode(labels, 2)
        assert result.dtype == np.float64

    def test_more_classes_than_labels(self):
        labels = np.array([0, 2])
        result = one_hot_encode(labels, 5)
        assert result.shape == (2, 5)
        assert result[0, 0] == 1.0
        assert result[1, 2] == 1.0
        assert result.sum() == 2.0

    def test_single_label(self):
        labels = np.array([3])
        result = one_hot_encode(labels, 5)
        expected = np.array([[0, 0, 0, 1, 0]], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_each_row_sums_to_one(self):
        labels = np.array([0, 1, 2, 3, 4, 0, 1])
        result = one_hot_encode(labels, 5)
        np.testing.assert_array_equal(result.sum(axis=1), np.ones(7))


# ---------------------------------------------------------------------------
# batch_dot_product
# ---------------------------------------------------------------------------

class TestBatchDotProduct:
    def test_basic(self):
        a = np.array([[1.0, 2.0, 3.0]])
        b = np.array([[4.0, 5.0, 6.0]])
        result = batch_dot_product(a, b)
        np.testing.assert_allclose(result, [32.0])

    def test_batch(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        b = np.array([[3.0, 4.0], [3.0, 4.0], [3.0, 4.0]])
        result = batch_dot_product(a, b)
        np.testing.assert_allclose(result, [3.0, 4.0, 7.0])

    def test_output_shape(self):
        a = np.random.randn(50, 10)
        b = np.random.randn(50, 10)
        result = batch_dot_product(a, b)
        assert result.shape == (50,)

    def test_orthogonal_vectors(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[0.0, 1.0], [1.0, 0.0]])
        result = batch_dot_product(a, b)
        np.testing.assert_allclose(result, [0.0, 0.0])

    def test_matches_numpy_einsum(self):
        a = np.random.randn(100, 64)
        b = np.random.randn(100, 64)
        result = batch_dot_product(a, b)
        expected = np.einsum("ij,ij->i", a, b)
        np.testing.assert_allclose(result, expected, atol=1e-10)
