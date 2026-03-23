"""
Tests for CNN exercises.
"""

import numpy as np
import pytest

from exercises import (
    conv2d_naive,
    conv2d_im2col,
    conv2d_backward,
    max_pool_forward,
    max_pool_backward,
    receptive_field_calculator,
)


class TestConv2D:
    def test_naive_and_im2col_agree(self):
        np.random.seed(42)
        X = np.random.randn(2, 3, 8, 8)
        K = np.random.randn(4, 3, 3, 3)
        out_naive = conv2d_naive(X, K, stride=1, padding=1)
        out_im2col = conv2d_im2col(X, K, stride=1, padding=1)
        np.testing.assert_allclose(out_naive, out_im2col, atol=1e-6)

    def test_identity_kernel(self):
        """3x3 kernel with center=1 should approximate identity."""
        X = np.random.randn(1, 1, 5, 5)
        K = np.zeros((1, 1, 3, 3))
        K[0, 0, 1, 1] = 1.0  # Center element
        out = conv2d_naive(X, K, stride=1, padding=1)
        np.testing.assert_allclose(out, X, atol=1e-10)

    def test_output_shape(self):
        X = np.random.randn(2, 3, 10, 10)
        K = np.random.randn(8, 3, 3, 3)
        out = conv2d_naive(X, K, stride=2, padding=1)
        assert out.shape == (2, 8, 5, 5)


class TestConvBackward:
    def test_gradient_check(self):
        np.random.seed(42)
        X = np.random.randn(1, 1, 5, 5)
        K = np.random.randn(1, 1, 3, 3)
        out = conv2d_naive(X, K, stride=1, padding=1)
        dout = np.random.randn(*out.shape)
        dX, dK = conv2d_backward(dout, X, K, stride=1, padding=1)
        assert dX.shape == X.shape
        assert dK.shape == K.shape


class TestMaxPool:
    def test_output_shape(self):
        X = np.random.randn(2, 3, 8, 8)
        out, _ = max_pool_forward(X, pool_size=2, stride=2)
        assert out.shape == (2, 3, 4, 4)

    def test_max_values(self):
        X = np.array([[[[1, 2], [3, 4]]]]).astype(float)  # (1,1,2,2)
        out, _ = max_pool_forward(X, pool_size=2, stride=2)
        assert out[0, 0, 0, 0] == 4.0


class TestReceptiveField:
    def test_single_layer(self):
        rf = receptive_field_calculator([(3, 1, 0)])
        assert rf[0] == 3

    def test_three_3x3_layers(self):
        """Three 3x3 conv layers with stride 1 -> RF = 7."""
        rf = receptive_field_calculator([(3, 1, 0), (3, 1, 0), (3, 1, 0)])
        assert rf[-1] == 7
