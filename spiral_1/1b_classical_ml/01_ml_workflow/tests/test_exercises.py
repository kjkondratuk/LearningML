"""
Tests for Module 01: ML Workflow

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    train_test_split_manual,
    compute_accuracy,
    compute_mse,
    compute_mae,
    detect_data_leakage,
)


# ---------------------------------------------------------------------------
# train_test_split_manual
# ---------------------------------------------------------------------------

class TestTrainTestSplit:
    def setup_method(self):
        self.X = np.arange(50).reshape(10, 5)
        self.y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    def test_split_sizes_default(self):
        X_train, X_test, y_train, y_test = train_test_split_manual(self.X, self.y)
        assert X_train.shape[0] == 8
        assert X_test.shape[0] == 2
        assert y_train.shape[0] == 8
        assert y_test.shape[0] == 2

    def test_split_sizes_custom(self):
        X_train, X_test, y_train, y_test = train_test_split_manual(
            self.X, self.y, test_size=0.3
        )
        assert X_train.shape[0] == 7
        assert X_test.shape[0] == 3

    def test_no_data_lost(self):
        X_train, X_test, y_train, y_test = train_test_split_manual(self.X, self.y)
        recombined = np.concatenate([X_train, X_test], axis=0)
        assert recombined.shape == self.X.shape
        # Every original row should appear exactly once
        original_sorted = np.sort(self.X, axis=0)
        recombined_sorted = np.sort(recombined, axis=0)
        np.testing.assert_array_equal(original_sorted, recombined_sorted)

    def test_shuffle_happens(self):
        X_train, X_test, _, _ = train_test_split_manual(self.X, self.y)
        # After shuffle, the first 8 rows should not be identical to X[:8]
        # (astronomically unlikely with seed 42 on 10 rows)
        is_same = np.array_equal(X_train, self.X[:8])
        assert not is_same, "Data does not appear to be shuffled"

    def test_reproducible(self):
        split1 = train_test_split_manual(self.X, self.y, random_state=7)
        split2 = train_test_split_manual(self.X, self.y, random_state=7)
        for a, b in zip(split1, split2):
            np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        split1 = train_test_split_manual(self.X, self.y, random_state=0)
        split2 = train_test_split_manual(self.X, self.y, random_state=99)
        assert not np.array_equal(split1[0], split2[0])

    def test_features_and_labels_stay_aligned(self):
        # Use a y that is a deterministic function of X so we can verify alignment
        X = np.arange(20).reshape(5, 4)
        y = X[:, 0]  # y equals the first column
        X_train, X_test, y_train, y_test = train_test_split_manual(
            X, y, test_size=0.4, random_state=1
        )
        np.testing.assert_array_equal(y_train, X_train[:, 0])
        np.testing.assert_array_equal(y_test, X_test[:, 0])


# ---------------------------------------------------------------------------
# compute_accuracy
# ---------------------------------------------------------------------------

class TestAccuracy:
    def test_perfect(self):
        y = np.array([0, 1, 1, 0])
        assert compute_accuracy(y, y) == 1.0

    def test_all_wrong(self):
        y_true = np.array([0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1])
        assert compute_accuracy(y_true, y_pred) == 0.0

    def test_partial(self):
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 0])
        assert compute_accuracy(y_true, y_pred) == pytest.approx(0.75)

    def test_returns_float(self):
        y = np.array([1, 0])
        result = compute_accuracy(y, y)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# compute_mse
# ---------------------------------------------------------------------------

class TestMSE:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_mse(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 5.0])
        # errors: 0, 0, 2 => squared: 0, 0, 4 => mean: 4/3
        assert compute_mse(y_true, y_pred) == pytest.approx(4.0 / 3.0)

    def test_symmetric(self):
        y_true = np.array([1.0, 3.0])
        y_pred = np.array([3.0, 1.0])
        # errors: -2, 2 => squared: 4, 4 => mean: 4
        assert compute_mse(y_true, y_pred) == pytest.approx(4.0)

    def test_single_element(self):
        assert compute_mse(np.array([5.0]), np.array([3.0])) == pytest.approx(4.0)

    def test_negative_values(self):
        y_true = np.array([-1.0, -2.0])
        y_pred = np.array([-1.0, -4.0])
        assert compute_mse(y_true, y_pred) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# compute_mae
# ---------------------------------------------------------------------------

class TestMAE:
    def test_zero_error(self):
        y = np.array([1.0, 2.0, 3.0])
        assert compute_mae(y, y) == pytest.approx(0.0)

    def test_known_value(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 5.0])
        assert compute_mae(y_true, y_pred) == pytest.approx(2.0 / 3.0)

    def test_symmetric(self):
        y_true = np.array([1.0, 3.0])
        y_pred = np.array([3.0, 1.0])
        assert compute_mae(y_true, y_pred) == pytest.approx(2.0)

    def test_negative_values(self):
        y_true = np.array([-1.0, -2.0])
        y_pred = np.array([-1.0, -4.0])
        assert compute_mae(y_true, y_pred) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# detect_data_leakage
# ---------------------------------------------------------------------------

class TestDetectDataLeakage:
    def test_no_leakage(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_test = np.array([[7, 8], [9, 10]])
        result = detect_data_leakage(X_train, X_test)
        assert result["has_leakage"] is False
        assert result["leaked_count"] == 0

    def test_full_leakage(self):
        X_train = np.array([[1, 2], [3, 4]])
        X_test = np.array([[1, 2], [3, 4]])
        result = detect_data_leakage(X_train, X_test)
        assert result["has_leakage"] is True
        assert result["leaked_count"] == 2

    def test_partial_leakage(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]])
        X_test = np.array([[3, 4], [7, 8]])
        result = detect_data_leakage(X_train, X_test)
        assert result["has_leakage"] is True
        assert result["leaked_count"] == 1

    def test_returns_dict(self):
        X_train = np.array([[1, 2]])
        X_test = np.array([[3, 4]])
        result = detect_data_leakage(X_train, X_test)
        assert isinstance(result, dict)
        assert "has_leakage" in result
        assert "leaked_count" in result
