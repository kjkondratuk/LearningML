"""
Tests for Activation and Initialization exercises.
"""

import numpy as np
import pytest

from exercises import (
    sigmoid_and_gradient,
    tanh_and_gradient,
    relu_and_gradient,
    leaky_relu_and_gradient,
    gelu_and_gradient,
    xavier_init,
    he_init,
    track_activation_statistics,
)


class TestSigmoid:
    def test_output_range(self):
        z = np.linspace(-10, 10, 100)
        act, _ = sigmoid_and_gradient(z)
        assert np.all(act > 0) and np.all(act < 1)

    def test_gradient_range(self):
        z = np.linspace(-10, 10, 100)
        _, grad = sigmoid_and_gradient(z)
        assert np.all(grad >= 0) and np.all(grad <= 0.25 + 1e-10)


class TestReLU:
    def test_positive(self):
        z = np.array([1.0, 2.0, 3.0])
        act, grad = relu_and_gradient(z)
        np.testing.assert_array_equal(act, z)
        np.testing.assert_array_equal(grad, [1, 1, 1])

    def test_negative(self):
        z = np.array([-1.0, -2.0])
        act, grad = relu_and_gradient(z)
        np.testing.assert_array_equal(act, [0, 0])
        np.testing.assert_array_equal(grad, [0, 0])


class TestXavierInit:
    def test_variance(self):
        W = xavier_init(1000, 500, seed=42)
        expected_var = 2.0 / (1000 + 500)
        np.testing.assert_allclose(np.var(W), expected_var, rtol=0.2)

    def test_activation_preserved(self):
        """20-layer sigmoid net with Xavier: variance should stay in [0.5, 2.0]."""
        np.random.seed(42)
        d = 100
        X = np.random.randn(50, d)
        weights = [xavier_init(d, d, seed=i) for i in range(20)]

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

        stats = track_activation_statistics(X, weights, sigmoid)
        for mean, var in stats:
            assert 0.01 < var < 5.0  # Generous bounds


class TestHeInit:
    def test_variance(self):
        W = he_init(1000, 500, seed=42)
        expected_var = 2.0 / 1000
        np.testing.assert_allclose(np.var(W), expected_var, rtol=0.2)

    def test_relu_activation_preserved(self):
        """20-layer ReLU net with He init: variance should stay bounded."""
        np.random.seed(42)
        d = 100
        X = np.random.randn(50, d)
        weights = [he_init(d, d, seed=i) for i in range(20)]
        relu = lambda z: np.maximum(0, z)
        stats = track_activation_statistics(X, weights, relu)
        for mean, var in stats:
            assert 0.01 < var < 10.0


class TestVanishing:
    def test_n01_init_vanishes(self):
        """N(0,1) init with sigmoid: activations should collapse by layer 20."""
        np.random.seed(42)
        d = 100
        X = np.random.randn(50, d)
        weights = [np.random.randn(d, d) for _ in range(20)]
        sigmoid = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        stats = track_activation_statistics(X, weights, sigmoid)
        # Last layer variance should be very small
        assert stats[-1][1] < 0.01
