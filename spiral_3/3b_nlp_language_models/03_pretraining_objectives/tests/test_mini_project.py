"""Tests for pretraining mini-project."""

import pytest
import torch

from mini_project import SmallTransformer


class TestSmallTransformer:
    def test_output_shape(self):
        model = SmallTransformer(vocab_size=1000, d_model=128, n_heads=4, n_layers=2)
        ids = torch.randint(0, 1000, (2, 32))
        out = model(ids)
        assert out.shape == (2, 32, 1000)
