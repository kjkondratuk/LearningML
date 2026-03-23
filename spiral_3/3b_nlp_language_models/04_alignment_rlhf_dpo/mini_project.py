"""
Mini-Project: DPO Fine-tuning on Preference Data

Fine-tune a small language model using DPO.

Reference: Rafailov et al. 2023 (arXiv:2305.18290)
"""

import torch
import torch.nn as nn


def prepare_preference_data(dataset_path: str) -> list[dict]:
    """Load and prepare preference pairs (chosen, rejected) for DPO training."""
    raise NotImplementedError


def compute_logprobs(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Compute per-token log probabilities under the model."""
    raise NotImplementedError


def train_dpo(
    model: nn.Module,
    ref_model: nn.Module,
    preference_data: list[dict],
    beta: float = 0.1,
    epochs: int = 3,
    lr: float = 1e-5,
) -> dict:
    """Train with DPO, returning loss history."""
    raise NotImplementedError
