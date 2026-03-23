"""Tests for Distributed Training."""
import pytest
import torch
from exercises import (
    all_reduce_ring, data_parallel_split, tensor_parallel_linear,
)

class TestAllReduce:
    def test_sum_equals_local(self):
        grads = [torch.randn(10) for _ in range(4)]
        result = all_reduce_ring(grads, world_size=4)
        expected = sum(grads)
        assert torch.allclose(result, expected, atol=1e-5)

class TestDataParallel:
    def test_correct_split(self):
        batch = torch.randn(8, 10)
        splits = data_parallel_split(batch, world_size=4)
        assert len(splits) == 4
        assert splits[0].shape[0] == 2

class TestTensorParallel:
    def test_matches_full_linear(self):
        torch.manual_seed(42)
        x = torch.randn(4, 8)
        W = torch.randn(8, 16)
        shards = [W[:, :8], W[:, 8:]]
        result = tensor_parallel_linear(x, shards, world_size=2)
        expected = x @ W
        assert torch.allclose(result, expected, atol=1e-5)
