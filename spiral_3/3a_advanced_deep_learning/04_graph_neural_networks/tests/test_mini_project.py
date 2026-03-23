"""Tests for GCN mini-project."""

import pytest
import torch

from mini_project import GCN


class TestGCN:
    def test_output_shape(self):
        model = GCN(in_features=1433, hidden_dim=16, num_classes=7)
        x = torch.randn(100, 1433)
        A = (torch.rand(100, 100) > 0.9).float()
        A = (A + A.T).clamp(max=1) - torch.eye(100)
        out = model(x, A)
        assert out.shape == (100, 7)
