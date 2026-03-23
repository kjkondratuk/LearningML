"""
Tests for Decision Trees and Ensembles exercises.
"""

import numpy as np
import pytest

from exercises import (
    gini_impurity,
    entropy_impurity,
    information_gain,
    build_decision_tree,
    predict_tree,
    bagging_ensemble,
    gradient_boosting,
    feature_importance,
)


class TestGiniImpurity:
    def test_pure_node(self):
        y = np.array([1, 1, 1, 1])
        np.testing.assert_allclose(gini_impurity(y), 0.0, atol=1e-10)

    def test_uniform_binary(self):
        """Gini of [0,0,1,1] = 1 - (0.5^2 + 0.5^2) = 0.5."""
        y = np.array([0, 0, 1, 1])
        np.testing.assert_allclose(gini_impurity(y), 0.5, atol=1e-10)

    def test_uniform_k_classes(self):
        """Gini of uniform over K classes = 1 - 1/K."""
        for K in [2, 3, 5, 10]:
            y = np.arange(K)
            expected = 1 - 1.0 / K
            np.testing.assert_allclose(gini_impurity(y), expected, atol=1e-10)


class TestEntropy:
    def test_fair_coin(self):
        y = np.array([0, 1])
        np.testing.assert_allclose(entropy_impurity(y), 1.0, atol=1e-10)

    def test_pure_node(self):
        y = np.array([1, 1, 1])
        np.testing.assert_allclose(entropy_impurity(y), 0.0, atol=1e-10)


class TestDecisionTree:
    def test_perfect_training_accuracy(self):
        """Unlimited tree should memorize training data."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = (X[:, 0] > 0).astype(int)
        tree = build_decision_tree(X, y, max_depth=None, min_samples=1)
        preds = predict_tree(tree, X)
        assert np.mean(preds == y) == 1.0

    def test_depth_limit(self):
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        tree = build_decision_tree(X, y, max_depth=2)
        preds = predict_tree(tree, X)
        assert np.mean(preds == y) > 0.7  # Should do okay


class TestRandomForest:
    def test_better_than_single_tree(self):
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 10)
        y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
        X_train, X_test = X[:150], X[150:]
        y_train, y_test = y[:150], y[150:]

        single_tree = build_decision_tree(X_train, y_train, max_depth=10)
        single_preds = predict_tree(single_tree, X_test)
        single_acc = np.mean(single_preds == y_test)

        forest = bagging_ensemble(X_train, y_train, n_trees=20, max_depth=10)
        # Majority vote
        all_preds = np.array([predict_tree(t, X_test) for t in forest])
        from scipy.stats import mode
        forest_preds = mode(all_preds, axis=0, keepdims=False).mode
        forest_acc = np.mean(forest_preds == y_test)

        assert forest_acc >= single_acc - 0.05  # Forest should be at least comparable


class TestGradientBoosting:
    def test_loss_decreases(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] + 0.5 * X[:, 1] + 0.1 * np.random.randn(100)
        trees, init_pred = gradient_boosting(X, y, n_rounds=20, lr=0.1, max_depth=3)

        # Compute MSE with boosting
        preds = np.full(len(y), init_pred)
        for tree in trees:
            preds += 0.1 * predict_tree(tree, X).astype(float)
        mse_boosted = np.mean((y - preds) ** 2)

        # MSE should be lower than just predicting the mean
        mse_mean = np.mean((y - np.mean(y)) ** 2)
        assert mse_boosted < mse_mean


class TestFeatureImportance:
    def test_identifies_informative_features(self):
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        trees = bagging_ensemble(X, y, n_trees=20, max_depth=5)
        imp = feature_importance(trees, 10)
        # Features 0 and 1 should have highest importance
        top2 = np.argsort(imp)[-2:]
        assert 0 in top2 or 1 in top2
