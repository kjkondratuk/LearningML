"""Tests for BDL exercises."""
import pytest
import torch
import torch.nn as nn
from exercises import (
    calibration_error, deep_ensemble_predict, mc_dropout_predict,
)

class TestMCDropout:
    def test_variance_shape(self):
        model = nn.Sequential(nn.Linear(10, 32), nn.Dropout(0.1), nn.Linear(32, 5))
        x = torch.randn(4, 10)
        mean, var = mc_dropout_predict(model, x, n_forward_passes=20)
        assert mean.shape == (4, 5)
        assert var.shape == (4, 5)
        assert (var >= 0).all()

class TestCalibration:
    def test_perfect_calibration(self):
        probs = torch.tensor([0.9, 0.1, 0.8, 0.2])
        labels = torch.tensor([1, 0, 1, 0])
        ece = calibration_error(probs, labels, n_bins=2)
        assert ece < 0.2
