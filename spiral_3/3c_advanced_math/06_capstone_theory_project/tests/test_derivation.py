"""Tests for generalization bound computation."""

import pytest
import torch
import torch.nn as nn

from derivation import (
    pac_bayes_bound,
    spectral_norm_bound,
)


class TestSpectralNormBound:
    def test_nonnegative(self):
        model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))
        bound = spectral_norm_bound(model, n_samples=1000)
        assert bound >= 0

    def test_increases_with_width(self):
        """Wider networks should have larger bounds."""
        model_narrow = nn.Sequential(nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 10))
        model_wide = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 10))
        bound_narrow = spectral_norm_bound(model_narrow, n_samples=1000)
        bound_wide = spectral_norm_bound(model_wide, n_samples=1000)
        assert bound_wide > bound_narrow


class TestPACBayesBound:
    def test_nonnegative(self):
        model = nn.Sequential(nn.Linear(10, 10))
        prior = nn.Sequential(nn.Linear(10, 10))
        bound = pac_bayes_bound(model, prior, n_samples=1000)
        assert bound >= 0
