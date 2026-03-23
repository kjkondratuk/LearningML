"""
Mini-Project: Node Classification on Cora

Build a 2-layer GCN for semi-supervised node classification on the Cora citation network.

Reference: Kipf & Welling 2016 (arXiv:1609.02907)
"""

import torch
import torch.nn as nn


class GCN(nn.Module):
    """Two-layer Graph Convolutional Network for node classification."""

    def __init__(self, in_features: int, hidden_dim: int, num_classes: int):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        """Forward pass: two GCN layers with ReLU and dropout.

        Returns log-softmax class probabilities for each node.
        """
        raise NotImplementedError


def train_gcn(model: GCN, features, adjacency, labels, train_mask, epochs: int = 200):
    """Train GCN with semi-supervised cross-entropy loss."""
    raise NotImplementedError
