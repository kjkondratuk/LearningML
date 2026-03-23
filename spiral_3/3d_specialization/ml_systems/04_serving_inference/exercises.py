"""Serving & Inference Exercises."""

import torch

def kv_cache_update(cache: dict, new_keys, new_values, layer_idx: int) -> dict:
    """Append new KV pairs for autoregressive generation."""
    raise NotImplementedError

def continuous_batching(requests: list, max_batch_size: int, max_seq_len: int) -> list:
    """Schedule requests with different lengths."""
    raise NotImplementedError

def speculative_decoding_step(draft_model, target_model, input_ids, gamma: int = 4):
    """Draft gamma tokens, verify with target model."""
    raise NotImplementedError

def onnx_export_model(pytorch_model, dummy_input, output_path: str):
    """Export and verify ONNX graph."""
    raise NotImplementedError
