"""
DDPM Evaluation: FID Score and Sample Quality Metrics

Reference: Heusel et al. 2017 (arXiv:1706.08500)
"""

import torch
import torch.nn as nn


def compute_fid(
    real_images: torch.Tensor,
    generated_images: torch.Tensor,
    feature_extractor: nn.Module | None = None,
) -> float:
    """Compute Frechet Inception Distance between real and generated images.

    Args:
        real_images: Real images, shape (N, C, H, W).
        generated_images: Generated images, shape (M, C, H, W).
        feature_extractor: Optional pretrained model for feature extraction.
                          If None, uses a simple CNN.

    Returns:
        FID score (lower is better).
    """
    raise NotImplementedError


def sample_and_evaluate(
    model: nn.Module,
    test_loader,
    n_samples: int = 10000,
    T: int = 1000,
) -> dict:
    """Generate samples and compute evaluation metrics.

    Returns:
        Dict with 'fid', 'sample_grid' (tensor for visualization).
    """
    raise NotImplementedError
