"""
Tests for Mini-Project: Classify Spirals

Run: pytest tests/test_mini_project.py -v
"""

import numpy as np
import pytest

from mini_project import generate_spirals, classify_spirals


class TestGenerateSpirals:
    def test_shapes(self):
        X, Y = generate_spirals(n_points=100, n_classes=2)
        assert X.shape == (2, 200)
        assert Y.shape == (2, 200)

    def test_one_hot(self):
        _, Y = generate_spirals(n_points=50, n_classes=3)
        # Each column should sum to 1
        np.testing.assert_array_equal(Y.sum(axis=0), np.ones(150))

    def test_reproducible(self):
        X1, Y1 = generate_spirals(seed=42)
        X2, Y2 = generate_spirals(seed=42)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(Y1, Y2)


class TestClassifySpirals:
    @pytest.mark.slow
    def test_accuracy_above_90(self):
        accuracy, loss_history = classify_spirals(
            hidden_size=100, lr=0.5, epochs=5000, seed=42
        )
        assert accuracy > 0.90, f"Expected >90% accuracy, got {accuracy:.1%}"

    def test_loss_decreases(self):
        _, loss_history = classify_spirals(epochs=500)
        assert loss_history[-1] < loss_history[0]
