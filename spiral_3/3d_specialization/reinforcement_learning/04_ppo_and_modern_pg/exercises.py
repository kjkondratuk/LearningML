"""PPO Exercises. Reference: Schulman et al. 2017 (arXiv:1707.06347)."""

import torch


def ppo_clip_loss(ratio: torch.Tensor, advantage: torch.Tensor, epsilon: float = 0.2) -> torch.Tensor:
    """PPO clipped surrogate: min(r*A, clip(r, 1-eps, 1+eps)*A).

    Returns scalar loss (negated for gradient ascent).
    """
    raise NotImplementedError


def trust_region_check(old_policy: torch.Tensor, new_policy: torch.Tensor, kl_threshold: float) -> bool:
    """Check KL(old || new) < threshold."""
    raise NotImplementedError


def compute_ratio(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor) -> torch.Tensor:
    """Importance sampling ratio: exp(new - old)."""
    raise NotImplementedError


def ppo_value_loss(predicted: torch.Tensor, target: torch.Tensor, old_predicted: torch.Tensor, clip_range: float = 0.2) -> torch.Tensor:
    """Clipped value function loss."""
    raise NotImplementedError
