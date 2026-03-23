"""
Tests for the Random Forest mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    predict_forest,
)
from exercises import (
    bagging_ensemble,
)


class TestPredictForest:
    def test_majority_vote(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        trees = bagging_ensemble(X, y, n_trees=10, max_depth=5)
        preds = predict_forest(trees, X)
        assert preds.shape == (100,)
        assert np.mean(preds == y) > 0.7
