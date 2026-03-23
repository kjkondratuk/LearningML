"""
Tests for Mini-Project: MNIST Digit Classifier

Run: pytest tests/test_mini_project.py -v

NOTE: These tests download MNIST data (~12MB) on first run.
"""

import pytest

from mini_project import mnist_digit_classifier


class TestMnistDigitClassifier:
    @pytest.mark.slow
    def test_accuracy_above_95(self):
        result = mnist_digit_classifier(
            hidden_dim=256, batch_size=64, lr=0.001, epochs=5
        )
        assert result["test_accuracy"] > 0.95, (
            f"Expected >95% accuracy, got {result['test_accuracy']:.1%}"
        )

    @pytest.mark.slow
    def test_loss_decreases(self):
        result = mnist_digit_classifier(epochs=3)
        losses = result["train_losses"]
        assert losses[-1] < losses[0]

    @pytest.mark.slow
    def test_returns_model(self):
        import torch.nn as nn
        result = mnist_digit_classifier(epochs=1)
        assert isinstance(result["model"], nn.Module)
