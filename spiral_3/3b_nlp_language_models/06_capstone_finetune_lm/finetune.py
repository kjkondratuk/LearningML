"""
LoRA/QLoRA Fine-tuning Pipeline

Implement LoRA from scratch, then fine-tune a 7B model.

Reference: Hu et al. 2021 (arXiv:2106.09685)
"""

import torch
import torch.nn as nn


class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer: W = W_0 + (alpha/r) * B @ A.

    Args:
        original_layer: The pretrained linear layer to adapt.
        rank: LoRA rank r.
        alpha: Scaling factor.
    """

    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: float = 16.0):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: original output + low-rank delta."""
        raise NotImplementedError


def apply_lora(model: nn.Module, rank: int = 16, target_modules: list[str] | None = None):
    """Apply LoRA to specified modules in the model."""
    raise NotImplementedError


def train_sft_with_lora(
    model: nn.Module,
    train_dataset,
    eval_dataset,
    epochs: int = 3,
    lr: float = 2e-4,
) -> dict:
    """Supervised fine-tuning with LoRA."""
    raise NotImplementedError
