"""
Tests for Capstone Model Definitions

Run: pytest tests/test_model.py -v
"""

import torch
import torch.nn as nn
import pytest

from model import SimpleCNN, TransferModel


# ---------------------------------------------------------------------------
# SimpleCNN
# ---------------------------------------------------------------------------

class TestSimpleCNN:
    def test_output_shape(self):
        model = SimpleCNN(num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_different_num_classes(self):
        model = SimpleCNN(num_classes=5)
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 5)

    def test_single_sample(self):
        model = SimpleCNN()
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 10)

    def test_gradients_flow(self):
        model = SimpleCNN()
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None

    def test_param_count_reasonable(self):
        """SimpleCNN should have between 100K and 5M parameters."""
        model = SimpleCNN()
        n_params = sum(p.numel() for p in model.parameters())
        assert 100_000 < n_params < 5_000_000

    def test_has_dropout(self):
        """Model should contain a Dropout layer."""
        model = SimpleCNN()
        has_dropout = any(
            isinstance(m, nn.Dropout) for m in model.modules()
        )
        assert has_dropout, "SimpleCNN should include Dropout"


# ---------------------------------------------------------------------------
# TransferModel
# ---------------------------------------------------------------------------

class TestTransferModel:
    @pytest.mark.slow
    def test_output_shape(self):
        model = TransferModel(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 10)

    @pytest.mark.slow
    def test_frozen_backbone(self):
        model = TransferModel(num_classes=10, freeze=True)
        frozen = 0
        trainable = 0
        for p in model.parameters():
            if p.requires_grad:
                trainable += p.numel()
            else:
                frozen += p.numel()
        # Backbone should be frozen (most params), only FC trainable
        assert frozen > trainable
        # FC layer: 512 * 10 + 10 = 5130
        assert trainable == 512 * 10 + 10

    @pytest.mark.slow
    def test_unfrozen_backbone(self):
        model = TransferModel(num_classes=10, freeze=False)
        all_trainable = all(p.requires_grad for p in model.parameters())
        assert all_trainable
