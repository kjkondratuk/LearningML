"""
Tests for Data Pipeline exercises.
"""

import numpy as np
import pytest

from exercises import (
    data_augmentation_pipeline,
    stratified_split,
    handle_imbalanced_data,
    data_validation,
)


class TestAugmentation:
    def test_output_shape(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        aug = data_augmentation_pipeline(img, seed=42)
        assert aug.shape == img.shape


class TestStratifiedSplit:
    def test_class_proportions(self):
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.array([0] * 100 + [1] * 60 + [2] * 40)
        X_tr, y_tr, X_val, y_val, X_te, y_te = stratified_split(X, y, 0.2, 0.1)
        # Class proportions should be within 5% of original
        orig_dist = np.bincount(y) / len(y)
        train_dist = np.bincount(y_tr) / len(y_tr)
        np.testing.assert_allclose(train_dist, orig_dist, atol=0.1)

    def test_no_overlap(self):
        X = np.arange(100).reshape(-1, 1)
        y = np.zeros(100, dtype=int)
        y[50:] = 1
        X_tr, _, X_val, _, X_te, _ = stratified_split(X, y, 0.2, 0.1)
        all_vals = np.concatenate([X_tr, X_val, X_te])
        assert len(np.unique(all_vals)) == 100


class TestImbalanced:
    def test_oversample_balances(self):
        np.random.seed(42)
        X = np.random.randn(110, 3)
        y = np.array([0] * 100 + [1] * 10)
        X_b, y_b = handle_imbalanced_data(X, y, "oversample")
        counts = np.bincount(y_b)
        assert counts[0] == counts[1]


class TestDataValidation:
    def test_detects_nans(self):
        X = np.array([[1, 2], [np.nan, 4], [5, 6]], dtype=float)
        y = np.array([0, 1, 0])
        report = data_validation(X, y)
        assert report['nan_count'] > 0

    def test_detects_constant_feature(self):
        X = np.array([[1, 5], [1, 6], [1, 7]], dtype=float)
        y = np.array([0, 1, 0])
        report = data_validation(X, y)
        assert 0 in report['constant_features']

    def test_detects_duplicates(self):
        X = np.array([[1, 2], [1, 2], [3, 4]], dtype=float)
        y = np.array([0, 0, 1])
        report = data_validation(X, y)
        assert report['duplicate_rows'] > 0
