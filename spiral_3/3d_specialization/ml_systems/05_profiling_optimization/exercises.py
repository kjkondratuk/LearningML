"""Profiling & Optimization Exercises."""

import torch

def profile_forward_pass(model, input, device: str = "cpu") -> dict:
    """Profile with PyTorch profiler, return time per layer."""
    raise NotImplementedError

def memory_estimate(model, batch_size: int, seq_len: int) -> dict:
    """Estimate peak GPU memory: params + activations + gradients."""
    raise NotImplementedError

def gradient_checkpointing_savings(model, num_layers: int) -> dict:
    """Compute memory savings: O(sqrt(L)) vs O(L)."""
    raise NotImplementedError

def mixed_precision_speedup(model, input) -> dict:
    """Compare FP32 vs FP16 throughput."""
    raise NotImplementedError
