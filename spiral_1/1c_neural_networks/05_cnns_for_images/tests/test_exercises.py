"""
Tests for Module 05: CNNs for Images

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import torch
import torch.nn as nn
import pytest

from exercises import (
    conv2d_manual,
    compute_output_size,
    build_cnn,
    count_parameters,
    train_cnn_on_cifar,
)


# ---------------------------------------------------------------------------
# conv2d_manual
# ---------------------------------------------------------------------------

class TestConv2dManual:
    def test_identity_kernel(self):
        """A centered 1 in a 1x1 kernel should return the image unchanged."""
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        kernel = np.array([[1.0]])
        result = conv2d_manual(image, kernel)
        np.testing.assert_array_equal(result, image)

    def test_known_convolution(self):
        image = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        kernel = np.array([[1, 0], [0, -1]], dtype=float)
        result = conv2d_manual(image, kernel)
        # Output should be 2x2
        assert result.shape == (2, 2)
        # Top-left: 1*1 + 2*0 + 4*0 + 5*(-1) = -4
        assert result[0, 0] == pytest.approx(-4.0)

    def test_output_shape(self):
        image = np.random.randn(10, 10)
        kernel = np.random.randn(3, 3)
        result = conv2d_manual(image, kernel)
        assert result.shape == (8, 8)

    def test_all_ones_kernel(self):
        """All-ones kernel computes local sum."""
        image = np.ones((4, 4))
        kernel = np.ones((2, 2))
        result = conv2d_manual(image, kernel)
        np.testing.assert_array_equal(result, np.full((3, 3), 4.0))

    def test_matches_numpy_correlate(self):
        """Compare with scipy if available, otherwise manual check."""
        image = np.random.randn(6, 6)
        kernel = np.random.randn(3, 3)
        result = conv2d_manual(image, kernel)
        # Manual verification of one element
        expected_00 = np.sum(image[:3, :3] * kernel)
        assert result[0, 0] == pytest.approx(expected_00, abs=1e-10)


# ---------------------------------------------------------------------------
# compute_output_size
# ---------------------------------------------------------------------------

class TestComputeOutputSize:
    def test_no_padding_stride_1(self):
        assert compute_output_size(32, 3, padding=0, stride=1) == 30

    def test_with_padding(self):
        assert compute_output_size(32, 3, padding=1, stride=1) == 32

    def test_with_stride(self):
        assert compute_output_size(32, 3, padding=1, stride=2) == 16

    def test_pooling(self):
        assert compute_output_size(32, 2, padding=0, stride=2) == 16

    def test_larger_kernel(self):
        assert compute_output_size(28, 5, padding=0, stride=1) == 24


# ---------------------------------------------------------------------------
# build_cnn
# ---------------------------------------------------------------------------

class TestBuildCnn:
    def test_output_shape(self):
        model = build_cnn(num_classes=10)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_different_num_classes(self):
        model = build_cnn(num_classes=5)
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 5)

    def test_is_nn_module(self):
        model = build_cnn()
        assert isinstance(model, nn.Module)


# ---------------------------------------------------------------------------
# count_parameters
# ---------------------------------------------------------------------------

class TestCountParameters:
    def test_simple_model(self):
        model = nn.Linear(10, 5)  # 10*5 + 5 = 55
        counts = count_parameters(model)
        assert counts["total"] == 55
        assert counts["trainable"] == 55

    def test_frozen_params(self):
        model = nn.Linear(10, 5)
        for p in model.parameters():
            p.requires_grad = False
        counts = count_parameters(model)
        assert counts["total"] == 55
        assert counts["trainable"] == 0

    def test_cnn(self):
        model = build_cnn()
        counts = count_parameters(model)
        assert counts["total"] > 0
        assert counts["trainable"] == counts["total"]


# ---------------------------------------------------------------------------
# train_cnn_on_cifar (slow test -- only run manually)
# ---------------------------------------------------------------------------

class TestTrainCnnOnCifar:
    @pytest.mark.slow
    def test_loss_decreases(self):
        model = build_cnn(num_classes=10)
        result = train_cnn_on_cifar(model, epochs=2, batch_size=128)
        assert result["train_losses"][-1] < result["train_losses"][0]

    @pytest.mark.slow
    def test_returns_expected_keys(self):
        model = build_cnn(num_classes=10)
        result = train_cnn_on_cifar(model, epochs=1)
        assert "train_losses" in result
        assert "test_accuracy" in result
