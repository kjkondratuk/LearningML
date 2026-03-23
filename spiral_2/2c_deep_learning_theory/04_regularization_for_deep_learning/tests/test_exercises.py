"""
Tests for DL Regularization exercises.
"""

import numpy as np
import pytest

from exercises import (
    dropout_forward,
    dropout_backward,
    batch_norm_forward,
    batch_norm_backward,
    layer_norm_forward,
)


class TestDropout:
    def test_zeros_fraction(self):
        """Dropout should zero approximately p fraction of neurons."""
        X = np.ones((1000, 100))
        out, mask = dropout_forward(X, p=0.5, training=True, seed=42)
        frac_zero = np.mean(out == 0)
        np.testing.assert_allclose(frac_zero, 0.5, atol=0.05)

    def test_eval_mode_identity(self):
        X = np.random.randn(10, 5)
        out, mask = dropout_forward(X, p=0.5, training=False)
        np.testing.assert_array_equal(out, X)
        assert mask is None

    def test_scaling(self):
        """Mean of output should approximately equal mean of input."""
        np.random.seed(42)
        X = np.random.randn(10000, 50)
        out, _ = dropout_forward(X, p=0.3, training=True, seed=42)
        np.testing.assert_allclose(out.mean(), X.mean(), atol=0.1)


class TestBatchNorm:
    def test_output_statistics(self):
        """Training mode: output should have ~zero mean, ~unit variance per feature."""
        np.random.seed(42)
        X = np.random.randn(64, 10) * 5 + 3
        gamma = np.ones(10)
        beta = np.zeros(10)
        rm = np.zeros(10)
        rv = np.ones(10)
        out, cache = batch_norm_forward(X, gamma, beta, rm, rv, training=True)
        np.testing.assert_allclose(out.mean(axis=0), 0, atol=0.1)
        np.testing.assert_allclose(out.var(axis=0), 1, atol=0.2)

    def test_running_stats_update(self):
        np.random.seed(42)
        X = np.random.randn(32, 5) * 3 + 2
        gamma = np.ones(5)
        beta = np.zeros(5)
        rm = np.zeros(5)
        rv = np.ones(5)
        batch_norm_forward(X, gamma, beta, rm, rv, training=True)
        # Running mean should have moved toward batch mean
        assert not np.allclose(rm, 0)

    def test_backward_gradient_check(self):
        np.random.seed(42)
        X = np.random.randn(8, 4)
        gamma = np.ones(4)
        beta = np.zeros(4)
        rm = np.zeros(4)
        rv = np.ones(4)
        out, cache = batch_norm_forward(X, gamma, beta, rm, rv, training=True)
        dout = np.random.randn(*out.shape)
        dx, dgamma, dbeta = batch_norm_backward(dout, cache)
        assert dx.shape == X.shape
        assert dgamma.shape == gamma.shape
        assert dbeta.shape == beta.shape


class TestLayerNorm:
    def test_normalizes_per_sample(self):
        np.random.seed(42)
        X = np.random.randn(10, 20) * 5 + 3
        gamma = np.ones(20)
        beta = np.zeros(20)
        out, _ = layer_norm_forward(X, gamma, beta)
        # Each sample should have ~zero mean and ~unit variance
        for i in range(10):
            np.testing.assert_allclose(out[i].mean(), 0, atol=0.1)
            np.testing.assert_allclose(out[i].var(), 1, atol=0.2)
