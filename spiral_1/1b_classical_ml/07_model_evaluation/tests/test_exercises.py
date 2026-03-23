"""
Tests for Module 07: Model Evaluation

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    confusion_matrix_manual,
    precision_recall_f1,
    roc_points,
    cross_validate_manual,
    learning_curve_data,
)


# ---------------------------------------------------------------------------
# confusion_matrix_manual
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        cm = confusion_matrix_manual(y_true, y_pred)
        expected = np.array([[2, 0], [0, 2]])
        np.testing.assert_array_equal(cm, expected)

    def test_all_wrong(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        cm = confusion_matrix_manual(y_true, y_pred)
        expected = np.array([[0, 2], [2, 0]])
        np.testing.assert_array_equal(cm, expected)

    def test_mixed(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        cm = confusion_matrix_manual(y_true, y_pred)
        # TN=2, FP=1, FN=1, TP=2
        expected = np.array([[2, 1], [1, 2]])
        np.testing.assert_array_equal(cm, expected)

    def test_output_shape(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        cm = confusion_matrix_manual(y_true, y_pred)
        assert cm.shape == (2, 2)

    def test_sums_to_n(self):
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0, 0, 1])
        cm = confusion_matrix_manual(y_true, y_pred)
        assert cm.sum() == 6


# ---------------------------------------------------------------------------
# precision_recall_f1
# ---------------------------------------------------------------------------

class TestPrecisionRecallF1:
    def test_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        result = precision_recall_f1(y_true, y_pred)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_no_true_positives(self):
        y_true = np.array([1, 1, 1])
        y_pred = np.array([0, 0, 0])
        result = precision_recall_f1(y_true, y_pred)
        assert result["recall"] == pytest.approx(0.0)

    def test_known_values(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        result = precision_recall_f1(y_true, y_pred)
        # TP=2, FP=1, FN=1
        assert result["precision"] == pytest.approx(2 / 3)
        assert result["recall"] == pytest.approx(2 / 3)
        assert result["f1"] == pytest.approx(2 / 3)

    def test_returns_dict(self):
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        result = precision_recall_f1(y_true, y_pred)
        assert isinstance(result, dict)
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result

    def test_high_precision_low_recall(self):
        y_true = np.array([1, 1, 1, 1, 0])
        y_pred = np.array([0, 0, 0, 1, 0])
        result = precision_recall_f1(y_true, y_pred)
        # TP=1, FP=0, FN=3
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# roc_points
# ---------------------------------------------------------------------------

class TestROCPoints:
    def test_output_shapes(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.4, 0.6, 0.9])
        fpr, tpr = roc_points(y_true, y_scores, n_thresholds=50)
        assert fpr.shape == (50,)
        assert tpr.shape == (50,)

    def test_fpr_sorted(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        y_scores = rng.rand(100)
        fpr, tpr = roc_points(y_true, y_scores)
        assert np.all(np.diff(fpr) >= -1e-10), "FPR should be sorted ascending"

    def test_range_zero_to_one(self):
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        y_scores = rng.rand(100)
        fpr, tpr = roc_points(y_true, y_scores)
        assert np.all(fpr >= 0) and np.all(fpr <= 1)
        assert np.all(tpr >= 0) and np.all(tpr <= 1)

    def test_perfect_model(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
        fpr, tpr = roc_points(y_true, y_scores, n_thresholds=50)
        # At some threshold, TPR should be ~1 and FPR should be ~0
        # Check that there exists a point with high TPR and low FPR
        good_points = (tpr > 0.9) & (fpr < 0.1)
        assert np.any(good_points)


# ---------------------------------------------------------------------------
# cross_validate_manual
# ---------------------------------------------------------------------------

class TestCrossValidateManual:
    def setup_method(self):
        rng = np.random.RandomState(42)
        self.X = rng.randn(100, 2)
        self.y = (self.X[:, 0] + self.X[:, 1] > 0).astype(int)

    def test_returns_k_scores(self):
        scores = cross_validate_manual(self.X, self.y, k=5)
        assert len(scores) == 5

    def test_three_folds(self):
        scores = cross_validate_manual(self.X, self.y, k=3)
        assert len(scores) == 3

    def test_scores_reasonable(self):
        scores = cross_validate_manual(self.X, self.y, k=5)
        for s in scores:
            assert 0.0 <= s <= 1.0
        assert np.mean(scores) > 0.7

    def test_scores_are_floats(self):
        scores = cross_validate_manual(self.X, self.y, k=5)
        for s in scores:
            assert isinstance(s, float)

    def test_reproducible(self):
        s1 = cross_validate_manual(self.X, self.y, k=5, random_state=0)
        s2 = cross_validate_manual(self.X, self.y, k=5, random_state=0)
        np.testing.assert_allclose(s1, s2)


# ---------------------------------------------------------------------------
# learning_curve_data
# ---------------------------------------------------------------------------

class TestLearningCurveData:
    def setup_method(self):
        rng = np.random.RandomState(42)
        self.X = rng.randn(200, 3)
        self.y = (self.X[:, 0] + 0.5 * self.X[:, 1] > 0).astype(int)

    def test_returns_dict(self):
        result = learning_curve_data(self.X, self.y)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = learning_curve_data(self.X, self.y)
        for key in ["train_sizes", "train_scores", "test_scores"]:
            assert key in result

    def test_default_sizes(self):
        result = learning_curve_data(self.X, self.y)
        assert len(result["train_sizes"]) == 6

    def test_custom_sizes(self):
        result = learning_curve_data(self.X, self.y, train_sizes=[0.2, 0.5, 1.0])
        assert len(result["train_sizes"]) == 3

    def test_train_sizes_increase(self):
        result = learning_curve_data(self.X, self.y)
        sizes = result["train_sizes"]
        for i in range(len(sizes) - 1):
            assert sizes[i] < sizes[i + 1]

    def test_scores_in_valid_range(self):
        result = learning_curve_data(self.X, self.y)
        for s in result["train_scores"] + result["test_scores"]:
            assert 0.0 <= s <= 1.0

    def test_train_score_high_for_small_data(self):
        result = learning_curve_data(self.X, self.y)
        # With very few training samples, a model can often memorize them
        # Train score at smallest size should be >= test score at smallest size
        assert result["train_scores"][0] >= result["test_scores"][0] - 0.1
