"""Tests for Serving & Inference."""
import pytest
from exercises import (
    kv_cache_update, continuous_batching, speculative_decoding_step, onnx_export_model,
)

class TestKVCache:
    def test_cache_grows(self):
        import torch
        cache = {"keys": [], "values": []}
        k = torch.randn(1, 4, 1, 32)
        v = torch.randn(1, 4, 1, 32)
        cache = kv_cache_update(cache, k, v, layer_idx=0)
        assert len(cache["keys"]) > 0
