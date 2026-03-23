"""
Tests for the MNIST multi-class classification mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    predict,
    confusion_matrix,
    accuracy,
)


class TestPredict:
    def test_argmax(self):
        W = np.eye(3)
        X = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        preds = predict(X, W)
        np.testing.assert_array_equal(preds, [0, 1, 2])


class TestConfusionMatrix:
    def test_perfect_predictions(self):
        y_true = np.array([0, 1, 2, 0, 1])
        y_pred = np.array([0, 1, 2, 0, 1])
        cm = confusion_matrix(y_true, y_pred, 3)
        expected = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 1]])
        np.testing.assert_array_equal(cm, expected)


class TestAccuracy:
    def test_perfect(self):
        assert accuracy(np.array([0, 1, 2]), np.array([0, 1, 2])) == 1.0

    def test_half(self):
        assert accuracy(np.array([0, 1, 0, 1]), np.array([0, 0, 0, 0])) == 0.5
