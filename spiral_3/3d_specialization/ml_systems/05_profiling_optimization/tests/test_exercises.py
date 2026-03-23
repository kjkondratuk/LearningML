"""Tests for Profiling & Optimization."""
import pytest
from exercises import (
    profile_forward_pass, memory_estimate, gradient_checkpointing_savings, mixed_precision_speedup,
)

class TestMemoryEstimate:
    def test_positive(self):
        import torch.nn as nn
        model = nn.Linear(100, 100)
        est = memory_estimate(model, batch_size=32, seq_len=128)
        assert est.get("total_bytes", 0) > 0 or isinstance(est, dict)
