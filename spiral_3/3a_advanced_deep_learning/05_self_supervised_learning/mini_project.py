"""
Mini-Project: SimCLR on CIFAR-10

Train a self-supervised representation using SimCLR, then evaluate
with linear probing.

Reference: Chen et al. 2020 (arXiv:2002.05709)
"""

import torch
import torch.nn as nn


class SimCLRModel(nn.Module):
    """SimCLR: encoder + projection head."""

    def __init__(self, base_encoder: nn.Module, projection_dim: int = 128):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (representation h, projection z)."""
        raise NotImplementedError


class LinearProbe(nn.Module):
    """Linear classifier on frozen representations."""

    def __init__(self, feature_dim: int, num_classes: int = 10):
        raise NotImplementedError

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def train_simclr(model: SimCLRModel, train_loader, epochs: int = 100, temperature: float = 0.1):
    """Train SimCLR encoder with NT-Xent loss."""
    raise NotImplementedError


def train_linear_probe(probe: LinearProbe, encoder: nn.Module, train_loader, epochs: int = 100):
    """Train linear probe on frozen encoder representations."""
    raise NotImplementedError
