"""Tests for SimCLR mini-project."""

import pytest
import torch
import torch.nn as nn

from mini_project import (
    LinearProbe,
    SimCLRModel,
)


class TestSimCLRModel:
    def test_output_shapes(self):
        encoder = nn.Sequential(nn.Flatten(), nn.Linear(3 * 32 * 32, 512))
        model = SimCLRModel(base_encoder=encoder, projection_dim=128)
        x = torch.randn(4, 3, 32, 32)
        h, z = model(x)
        assert h.shape[0] == 4
        assert z.shape == (4, 128)


class TestLinearProbe:
    def test_output_shape(self):
        probe = LinearProbe(feature_dim=512, num_classes=10)
        h = torch.randn(8, 512)
        out = probe(h)
        assert out.shape == (8, 10)
