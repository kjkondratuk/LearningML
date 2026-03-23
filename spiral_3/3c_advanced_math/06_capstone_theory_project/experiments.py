"""
Empirical Validation of Generalization Bounds

Train networks of varying width and compare bounds to actual test error.
"""

import torch
import torch.nn as nn


def train_varying_width(widths: list[int], dataset, epochs: int = 50) -> dict:
    """Train networks of varying width on MNIST."""
    raise NotImplementedError


def plot_bound_vs_error(widths, bounds, test_errors, train_errors):
    """Plot generalization bounds vs actual test error."""
    raise NotImplementedError
