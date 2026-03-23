"""
Tests for Module 04: The Training Loop

Run: pytest tests/test_exercises.py -v
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytest

from exercises import (
    create_dataset,
    create_dataloader,
    build_classifier,
    train_one_epoch,
    evaluate,
    full_training_loop,
)


# ---------------------------------------------------------------------------
# create_dataset
# ---------------------------------------------------------------------------

class TestCreateDataset:
    def test_length(self):
        ds = create_dataset(n_samples=100)
        assert len(ds) == 100

    def test_getitem_returns_tuple(self):
        ds = create_dataset(n_samples=10)
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, (torch.Tensor, int))

    def test_feature_dimension(self):
        ds = create_dataset(n_samples=10, n_features=5)
        x, _ = ds[0]
        assert x.shape == (5,)

    def test_labels_in_range(self):
        ds = create_dataset(n_samples=100, n_classes=4)
        for i in range(len(ds)):
            _, y = ds[i]
            label = y.item() if isinstance(y, torch.Tensor) else y
            assert 0 <= label < 4


# ---------------------------------------------------------------------------
# create_dataloader
# ---------------------------------------------------------------------------

class TestCreateDataloader:
    def test_returns_dataloader(self):
        ds = create_dataset(n_samples=100)
        dl = create_dataloader(ds, batch_size=16)
        assert isinstance(dl, DataLoader)

    def test_batch_size(self):
        ds = create_dataset(n_samples=100)
        dl = create_dataloader(ds, batch_size=16, shuffle=False)
        batch_x, batch_y = next(iter(dl))
        assert batch_x.shape[0] == 16


# ---------------------------------------------------------------------------
# build_classifier
# ---------------------------------------------------------------------------

class TestBuildClassifier:
    def test_output_shape(self):
        model = build_classifier(10, 64, 3)
        x = torch.randn(8, 10)
        out = model(x)
        assert out.shape == (8, 3)

    def test_is_nn_module(self):
        model = build_classifier(10, 64, 3)
        assert isinstance(model, nn.Module)

    def test_has_parameters(self):
        model = build_classifier(10, 64, 3)
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0


# ---------------------------------------------------------------------------
# train_one_epoch
# ---------------------------------------------------------------------------

class TestTrainOneEpoch:
    def test_returns_float(self):
        ds = create_dataset(n_samples=100, n_features=10, n_classes=3)
        dl = create_dataloader(ds, batch_size=32)
        model = build_classifier(10, 32, 3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss = train_one_epoch(model, dl, criterion, optimizer)
        assert isinstance(loss, float)
        assert loss > 0

    def test_loss_decreases_over_epochs(self):
        ds = create_dataset(n_samples=200, n_features=10, n_classes=3, seed=0)
        dl = create_dataloader(ds, batch_size=32)
        model = build_classifier(10, 64, 3)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        losses = []
        for _ in range(10):
            losses.append(train_one_epoch(model, dl, criterion, optimizer))
        assert losses[-1] < losses[0]


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_float_in_range(self):
        ds = create_dataset(n_samples=100, n_features=10, n_classes=3)
        dl = create_dataloader(ds, batch_size=32, shuffle=False)
        model = build_classifier(10, 32, 3)
        acc = evaluate(model, dl)
        assert isinstance(acc, float)
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# full_training_loop
# ---------------------------------------------------------------------------

class TestFullTrainingLoop:
    def test_returns_expected_keys(self):
        result = full_training_loop(
            n_samples=200, epochs=3, hidden_dim=32
        )
        assert "train_losses" in result
        assert "test_accuracy" in result
        assert "model" in result

    def test_train_losses_length(self):
        result = full_training_loop(n_samples=200, epochs=5, hidden_dim=32)
        assert len(result["train_losses"]) == 5

    def test_accuracy_above_random(self):
        """After training, accuracy should beat random chance (1/3 ~ 0.33)."""
        result = full_training_loop(
            n_samples=500, epochs=20, hidden_dim=64, lr=0.001
        )
        assert result["test_accuracy"] > 0.4
