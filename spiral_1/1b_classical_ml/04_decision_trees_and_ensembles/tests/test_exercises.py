"""
Tests for Module 04: Decision Trees & Ensembles

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    gini_impurity,
    information_gain,
    find_best_split,
    fit_decision_stump,
    sklearn_random_forest,
    sklearn_gradient_boosting,
)


# ---------------------------------------------------------------------------
# gini_impurity
# ---------------------------------------------------------------------------

class TestGiniImpurity:
    def test_pure_node(self):
        y = np.array([1, 1, 1, 1])
        assert gini_impurity(y) == pytest.approx(0.0)

    def test_maximum_binary_impurity(self):
        y = np.array([0, 0, 1, 1])
        assert gini_impurity(y) == pytest.approx(0.5)

    def test_three_classes_equal(self):
        y = np.array([0, 1, 2, 0, 1, 2])
        expected = 1 - (1 / 3) ** 2 - (1 / 3) ** 2 - (1 / 3) ** 2
        assert gini_impurity(y) == pytest.approx(expected)

    def test_imbalanced(self):
        y = np.array([0, 0, 0, 1])
        expected = 1 - (3 / 4) ** 2 - (1 / 4) ** 2
        assert gini_impurity(y) == pytest.approx(expected)

    def test_single_element(self):
        y = np.array([0])
        assert gini_impurity(y) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# information_gain
# ---------------------------------------------------------------------------

class TestInformationGain:
    def test_perfect_split(self):
        y_parent = np.array([0, 0, 1, 1])
        y_left = np.array([0, 0])
        y_right = np.array([1, 1])
        ig = information_gain(y_parent, y_left, y_right)
        assert ig == pytest.approx(0.5)

    def test_no_gain(self):
        y_parent = np.array([0, 1, 0, 1])
        y_left = np.array([0, 1])
        y_right = np.array([0, 1])
        ig = information_gain(y_parent, y_left, y_right)
        assert ig == pytest.approx(0.0)

    def test_partial_gain(self):
        y_parent = np.array([0, 0, 1, 1])
        y_left = np.array([0, 0, 1])
        y_right = np.array([1])
        ig = information_gain(y_parent, y_left, y_right)
        assert ig > 0
        assert ig < 0.5

    def test_non_negative(self):
        rng = np.random.RandomState(42)
        y = rng.randint(0, 3, 50)
        split = rng.randint(10, 40)
        ig = information_gain(y, y[:split], y[split:])
        assert ig >= -1e-10  # allow tiny floating point error


# ---------------------------------------------------------------------------
# find_best_split
# ---------------------------------------------------------------------------

class TestFindBestSplit:
    def test_obvious_split(self):
        # Feature 0 perfectly separates the classes
        X = np.array([[1.0, 5.0], [2.0, 6.0], [3.0, 7.0],
                       [10.0, 5.0], [11.0, 6.0], [12.0, 7.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        result = find_best_split(X, y)
        assert result["feature_index"] == 0
        assert 3.0 < result["threshold"] < 10.0
        assert result["info_gain"] > 0.4

    def test_returns_dict(self):
        X = np.array([[1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1])
        result = find_best_split(X, y)
        assert "feature_index" in result
        assert "threshold" in result
        assert "info_gain" in result

    def test_second_feature_better(self):
        # Feature 1 separates, feature 0 does not
        X = np.array([[1.0, 1.0], [2.0, 2.0], [1.0, 10.0], [2.0, 11.0]])
        y = np.array([0, 0, 1, 1])
        result = find_best_split(X, y)
        assert result["feature_index"] == 1


# ---------------------------------------------------------------------------
# fit_decision_stump
# ---------------------------------------------------------------------------

class TestFitDecisionStump:
    def test_simple_stump(self):
        X = np.array([[1.0], [2.0], [3.0], [10.0], [11.0], [12.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        stump = fit_decision_stump(X, y)
        assert stump["feature_index"] == 0
        assert stump["left_label"] == 0
        assert stump["right_label"] == 1

    def test_returns_required_keys(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        stump = fit_decision_stump(X, y)
        for key in ["feature_index", "threshold", "left_label", "right_label"]:
            assert key in stump

    def test_stump_predictions_reasonable(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)
        stump = fit_decision_stump(X, y)

        # Predict using stump
        feature = X[:, stump["feature_index"]]
        preds = np.where(
            feature <= stump["threshold"],
            stump["left_label"],
            stump["right_label"],
        )
        accuracy = np.mean(preds == y)
        assert accuracy > 0.7


# ---------------------------------------------------------------------------
# sklearn_random_forest
# ---------------------------------------------------------------------------

class TestSklearnRandomForest:
    def setup_method(self):
        rng = np.random.RandomState(42)
        self.X = rng.randn(200, 4)
        self.y = ((self.X[:, 0] + self.X[:, 1]) > 0).astype(int)
        self.X_train, self.X_test = self.X[:160], self.X[160:]
        self.y_train, self.y_test = self.y[:160], self.y[160:]

    def test_output_shape(self):
        y_pred, train_acc = sklearn_random_forest(
            self.X_train, self.y_train, self.X_test
        )
        assert y_pred.shape == (40,)

    def test_train_accuracy_high(self):
        _, train_acc = sklearn_random_forest(
            self.X_train, self.y_train, self.X_test
        )
        assert train_acc > 0.9

    def test_reasonable_predictions(self):
        y_pred, _ = sklearn_random_forest(
            self.X_train, self.y_train, self.X_test
        )
        test_acc = np.mean(y_pred == self.y_test)
        assert test_acc > 0.7


# ---------------------------------------------------------------------------
# sklearn_gradient_boosting
# ---------------------------------------------------------------------------

class TestSklearnGradientBoosting:
    def setup_method(self):
        rng = np.random.RandomState(42)
        self.X = rng.randn(200, 4)
        self.y = ((self.X[:, 0] + self.X[:, 1]) > 0).astype(int)
        self.X_train, self.X_test = self.X[:160], self.X[160:]
        self.y_train, self.y_test = self.y[:160], self.y[160:]

    def test_output_shape(self):
        y_pred, train_acc = sklearn_gradient_boosting(
            self.X_train, self.y_train, self.X_test
        )
        assert y_pred.shape == (40,)

    def test_train_accuracy_high(self):
        _, train_acc = sklearn_gradient_boosting(
            self.X_train, self.y_train, self.X_test
        )
        assert train_acc > 0.9

    def test_reasonable_predictions(self):
        y_pred, _ = sklearn_gradient_boosting(
            self.X_train, self.y_train, self.X_test
        )
        test_acc = np.mean(y_pred == self.y_test)
        assert test_acc > 0.7
