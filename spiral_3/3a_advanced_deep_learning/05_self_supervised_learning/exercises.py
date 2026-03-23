"""
Self-Supervised Learning Exercises

Implement SSL components from:
    Chen et al. (2020) SimCLR (arXiv:2002.05709)
    He et al. (2021) MAE (arXiv:2111.06377)
    Grill et al. (2020) BYOL (arXiv:2006.07733)
"""

import torch


def info_nce_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negatives: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """InfoNCE contrastive loss.

    L = -log(exp(sim(a, p) / tau) / (exp(sim(a, p) / tau) + sum exp(sim(a, n_k) / tau)))

    where sim is cosine similarity.

    Args:
        anchor: Anchor embeddings, shape (batch_size, embed_dim).
        positive: Positive pair embeddings, shape (batch_size, embed_dim).
        negatives: Negative embeddings, shape (batch_size, num_negatives, embed_dim).
        temperature: Temperature scaling parameter.

    Returns:
        Scalar loss (mean over batch).
    """
    raise NotImplementedError


def simclr_augmentation_pair(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate two augmented views of an image for SimCLR.

    Apply random composition of: random crop + resize, color jitter,
    random horizontal flip, Gaussian blur, and grayscale conversion.

    Args:
        image: Input image tensor, shape (C, H, W).

    Returns:
        Tuple of two augmented views, each shape (C, H, W).
    """
    raise NotImplementedError


def nt_xent_loss(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Normalized Temperature-scaled Cross-Entropy loss (SimCLR).

    Given a batch of N pairs (z_i, z_j), compute the NT-Xent loss.
    For each positive pair (i, j), all other 2(N-1) samples serve as negatives.

    Args:
        z_i: First view embeddings, shape (N, embed_dim).
        z_j: Second view embeddings, shape (N, embed_dim).
        temperature: Temperature parameter.

    Returns:
        Scalar loss.
    """
    raise NotImplementedError


def masked_autoencoder_loss(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """MAE loss: MSE only on masked patches.

    Args:
        original: Original patches, shape (batch_size, num_patches, patch_dim).
        reconstructed: Reconstructed patches, shape (batch_size, num_patches, patch_dim).
        mask: Binary mask, shape (batch_size, num_patches). 1 = masked (compute loss).

    Returns:
        Scalar loss (mean over masked patches).
    """
    raise NotImplementedError


def byol_loss(
    online_projection: torch.Tensor,
    target_projection: torch.Tensor,
) -> torch.Tensor:
    """BYOL loss: negative cosine similarity between online and target projections.

    loss = 2 - 2 * cos_sim(online, target)

    Note: stop-gradient is applied to the target externally.

    Args:
        online_projection: Online network output, shape (batch_size, embed_dim).
        target_projection: Target network output (detached), shape (batch_size, embed_dim).

    Returns:
        Scalar loss.
    """
    raise NotImplementedError
