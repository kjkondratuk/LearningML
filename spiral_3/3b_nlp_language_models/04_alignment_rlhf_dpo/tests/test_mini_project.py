"""Tests for DPO mini-project."""

import pytest
import torch
import torch.nn as nn

from mini_project import compute_logprobs


class TestComputeLogprobs:
    def test_output_shape(self):
        model = nn.Sequential(nn.Embedding(100, 64), nn.Linear(64, 100))
        ids = torch.randint(0, 100, (2, 10))
        logprobs = compute_logprobs(model, ids)
        assert logprobs.shape[0] == 2
