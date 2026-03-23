"""
Tests for Module 04: Mini-Project — Compare Tree Depths

Run: pytest tests/test_mini_project.py -v
"""

import numpy as np
import pytest

from mini_project import compare_tree_depths


class TestCompareTreeDepths:
    def setup_method(self):
        rng = np.random.RandomState(42)
        self.X = rng.randn(300, 5)
        # Non-linear decision boundary to make depth matter
        self.y = (
            (self.X[:, 0] ** 2 + self.X[:, 1] ** 2 > 1.0)
        ).astype(int)

    def test_returns_dict(self):
        result = compare_tree_depths(self.X, self.y)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = compare_tree_depths(self.X, self.y)
        for key in ["depths", "train_accs", "test_accs", "best_depth"]:
            assert key in result

    def test_default_depths(self):
        result = compare_tree_depths(self.X, self.y)
        assert result["depths"] == [1, 2, 3, 5, 10, 20, None]

    def test_custom_depths(self):
        result = compare_tree_depths(self.X, self.y, depths=[1, 5, None])
        assert result["depths"] == [1, 5, None]
        assert len(result["train_accs"]) == 3
        assert len(result["test_accs"]) == 3

    def test_train_acc_increases_with_depth(self):
        result = compare_tree_depths(self.X, self.y)
        # Training accuracy should generally increase with depth
        # The unlimited depth tree should have highest train accuracy
        assert result["train_accs"][-1] >= result["train_accs"][0]

    def test_overfitting_gap(self):
        result = compare_tree_depths(self.X, self.y)
        # For unlimited depth, train acc should be higher than test acc
        # (evidence of overfitting)
        assert result["train_accs"][-1] > result["test_accs"][-1]

    def test_best_depth_is_valid(self):
        result = compare_tree_depths(self.X, self.y)
        assert result["best_depth"] in result["depths"]

    def test_best_depth_has_highest_test_acc(self):
        result = compare_tree_depths(self.X, self.y)
        best_idx = result["depths"].index(result["best_depth"])
        assert result["test_accs"][best_idx] == max(result["test_accs"])
