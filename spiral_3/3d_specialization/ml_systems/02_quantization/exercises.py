"""Quantization Exercises. References: Frantar et al. 2022 (arXiv:2210.17323), Lin et al. 2023 (arXiv:2306.00978)."""

import torch


def symmetric_quantize(tensor: torch.Tensor, num_bits: int = 8) -> tuple[torch.Tensor, float]:
    """Symmetric quantization: q = round(x / scale)."""
    raise NotImplementedError


def asymmetric_quantize(tensor: torch.Tensor, num_bits: int = 8) -> tuple[torch.Tensor, float, int]:
    """Asymmetric quantization with zero point."""
    raise NotImplementedError


def dequantize(quantized: torch.Tensor, scale: float, zero_point: int = 0) -> torch.Tensor:
    """Reconstruct: x_approx = (q - zero_point) * scale."""
    raise NotImplementedError


def quantization_error(original: torch.Tensor, quantized_dequantized: torch.Tensor) -> dict[str, float]:
    """Compute MSE and max absolute error."""
    raise NotImplementedError


def mixed_precision_forward(model, input: torch.Tensor, sensitive_layers: list[str]) -> torch.Tensor:
    """FP16 for most layers, FP32 for sensitive layers."""
    raise NotImplementedError
