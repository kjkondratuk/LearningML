"""
Tests for Module 01: Perceptron & Neurons

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    step_function,
    perceptron_predict,
    perceptron_train,
    relu,
    softmax,
)


# ---------------------------------------------------------------------------
# step_function
# ---------------------------------------------------------------------------

class TestStepFunction:
    def test_positive(self):
        assert step_function(1.0) == 1

    def test_negative(self):
        assert step_function(-0.5) == 0

    def test_zero(self):
        assert step_function(0.0) == 1

    def test_large_positive(self):
        assert step_function(1e6) == 1

    def test_large_negative(self):
        assert step_function(-1e6) == 0


# ---------------------------------------------------------------------------
# perceptron_predict
# ---------------------------------------------------------------------------

class TestPerceptronPredict:
    def test_and_gate_true(self):
        w = np.array([1.0, 1.0])
        assert perceptron_predict(w, -1.5, np.array([1.0, 1.0])) == 1

    def test_and_gate_false(self):
        w = np.array([1.0, 1.0])
        assert perceptron_predict(w, -1.5, np.array([1.0, 0.0])) == 0
        assert perceptron_predict(w, -1.5, np.array([0.0, 1.0])) == 0
        assert perceptron_predict(w, -1.5, np.array([0.0, 0.0])) == 0

    def test_or_gate(self):
        w = np.array([1.0, 1.0])
        assert perceptron_predict(w, -0.5, np.array([0.0, 0.0])) == 0
        assert perceptron_predict(w, -0.5, np.array([1.0, 0.0])) == 1
        assert perceptron_predict(w, -0.5, np.array([0.0, 1.0])) == 1
        assert perceptron_predict(w, -0.5, np.array([1.0, 1.0])) == 1


# ---------------------------------------------------------------------------
# perceptron_train
# ---------------------------------------------------------------------------

class TestPerceptronTrain:
    def test_learns_and_gate(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 0, 0, 1])
        w, b = perceptron_train(X, y, lr=0.1, max_epochs=100)
        for xi, yi in zip(X, y):
            assert perceptron_predict(w, b, xi) == yi

    def test_learns_or_gate(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 1])
        w, b = perceptron_train(X, y, lr=0.1, max_epochs=100)
        for xi, yi in zip(X, y):
            assert perceptron_predict(w, b, xi) == yi

    def test_returns_correct_shapes(self):
        X = np.array([[0, 0], [1, 1]], dtype=float)
        y = np.array([0, 1])
        w, b = perceptron_train(X, y)
        assert w.shape == (2,)
        assert isinstance(b, (float, np.floating))

    def test_xor_does_not_converge(self):
        """XOR is not linearly separable -- the perceptron should fail."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0])
        w, b = perceptron_train(X, y, lr=0.1, max_epochs=200)
        predictions = [perceptron_predict(w, b, xi) for xi in X]
        # At least one sample should be wrong
        assert predictions != list(y), (
            "A single perceptron should not be able to learn XOR"
        )


# ---------------------------------------------------------------------------
# relu
# ---------------------------------------------------------------------------

class TestRelu:
    def test_positive_unchanged(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(relu(x), x)

    def test_negative_clipped(self):
        x = np.array([-1.0, -2.0, -3.0])
        np.testing.assert_array_equal(relu(x), np.zeros(3))

    def test_mixed(self):
        x = np.array([-2.0, 0.0, 3.0])
        expected = np.array([0.0, 0.0, 3.0])
        np.testing.assert_array_equal(relu(x), expected)

    def test_2d_array(self):
        x = np.array([[-1, 2], [3, -4]], dtype=float)
        expected = np.array([[0, 2], [3, 0]], dtype=float)
        np.testing.assert_array_equal(relu(x), expected)


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_sums_to_one(self):
        z = np.array([1.0, 2.0, 3.0])
        result = softmax(z)
        assert abs(result.sum() - 1.0) < 1e-7

    def test_all_positive(self):
        z = np.array([1.0, 2.0, 3.0])
        result = softmax(z)
        assert np.all(result > 0)

    def test_order_preserved(self):
        z = np.array([1.0, 2.0, 3.0])
        result = softmax(z)
        assert result[2] > result[1] > result[0]

    def test_equal_inputs(self):
        z = np.array([1.0, 1.0, 1.0])
        result = softmax(z)
        np.testing.assert_allclose(result, np.ones(3) / 3, atol=1e-7)

    def test_numerical_stability(self):
        """Large values should not cause overflow."""
        z = np.array([1000.0, 1001.0, 1002.0])
        result = softmax(z)
        assert np.all(np.isfinite(result))
        assert abs(result.sum() - 1.0) < 1e-7

    def test_known_values(self):
        z = np.array([0.0, 0.0])
        result = softmax(z)
        np.testing.assert_allclose(result, np.array([0.5, 0.5]), atol=1e-7)
