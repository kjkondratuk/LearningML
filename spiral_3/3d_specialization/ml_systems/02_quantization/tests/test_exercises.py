"""Tests for Quantization."""
import pytest
import torch
from exercises import (
    dequantize, quantization_error, symmetric_quantize,
)

class TestSymmetricQuantize:
    def test_roundtrip_error_bounded(self):
        t = torch.randn(100)
        q, scale = symmetric_quantize(t, num_bits=8)
        t_approx = dequantize(q, scale)
        err = quantization_error(t, t_approx)
        assert err["max_error"] < scale / 2 + 1e-6

    def test_8bit_low_error(self):
        t = torch.randn(1000)
        q, scale = symmetric_quantize(t, num_bits=8)
        t_approx = dequantize(q, scale)
        err = quantization_error(t, t_approx)
        relative_err = err["mse"] / t.var().item()
        assert relative_err < 0.01
