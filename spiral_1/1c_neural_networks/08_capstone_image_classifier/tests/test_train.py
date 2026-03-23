"""
Tests for Capstone Training Pipeline

Run: pytest tests/test_train.py -v

NOTE: Some tests download CIFAR-10 (~170MB) on first run.
"""

import torch.nn as nn
import pytest

from train import load_data, train, evaluate_and_report, compare_models
from model import SimpleCNN


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

class TestLoadData:
    @pytest.mark.slow
    def test_returns_expected_keys(self):
        data = load_data(batch_size=16, resize_for_transfer=False)
        assert "train_loader" in data
        assert "test_loader" in data
        assert "classes" in data

    @pytest.mark.slow
    def test_batch_shape_simple(self):
        data = load_data(batch_size=8, resize_for_transfer=False)
        batch_x, batch_y = next(iter(data["train_loader"]))
        assert batch_x.shape == (8, 3, 32, 32)
        assert batch_y.shape == (8,)

    @pytest.mark.slow
    def test_batch_shape_transfer(self):
        data = load_data(batch_size=4, resize_for_transfer=True)
        batch_x, _ = next(iter(data["train_loader"]))
        assert batch_x.shape[2] == 224
        assert batch_x.shape[3] == 224

    @pytest.mark.slow
    def test_ten_classes(self):
        data = load_data()
        assert len(data["classes"]) == 10


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

class TestTrain:
    @pytest.mark.slow
    def test_loss_decreases(self):
        model = SimpleCNN(num_classes=10)
        data = load_data(batch_size=128, resize_for_transfer=False)
        result = train(model, data["train_loader"], epochs=2, lr=0.001)
        assert result["train_losses"][-1] < result["train_losses"][0]

    @pytest.mark.slow
    def test_returns_expected_keys(self):
        model = SimpleCNN(num_classes=10)
        data = load_data(batch_size=128, resize_for_transfer=False)
        result = train(model, data["train_loader"], epochs=1, lr=0.001)
        assert "train_losses" in result
        assert "train_accuracies" in result


# ---------------------------------------------------------------------------
# evaluate_and_report
# ---------------------------------------------------------------------------

class TestEvaluateAndReport:
    @pytest.mark.slow
    def test_returns_expected_keys(self):
        model = SimpleCNN(num_classes=10)
        data = load_data(batch_size=128, resize_for_transfer=False)
        result = evaluate_and_report(model, data["test_loader"])
        assert "test_accuracy" in result
        assert "per_class_accuracy" in result
        assert "confusion_matrix" in result

    @pytest.mark.slow
    def test_accuracy_in_range(self):
        model = SimpleCNN(num_classes=10)
        data = load_data(batch_size=128, resize_for_transfer=False)
        result = evaluate_and_report(model, data["test_loader"])
        assert 0.0 <= result["test_accuracy"] <= 1.0

    @pytest.mark.slow
    def test_confusion_matrix_shape(self):
        model = SimpleCNN(num_classes=10)
        data = load_data(batch_size=128, resize_for_transfer=False)
        result = evaluate_and_report(model, data["test_loader"])
        assert result["confusion_matrix"].shape == (10, 10)


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:
    def test_returns_string(self):
        simple = {"test_accuracy": 0.72, "per_class_accuracy": {"cat": 0.6}}
        transfer = {"test_accuracy": 0.88, "per_class_accuracy": {"cat": 0.85}}
        result = compare_models(simple, transfer)
        assert isinstance(result, str)
        assert "0.72" in result or "72" in result
