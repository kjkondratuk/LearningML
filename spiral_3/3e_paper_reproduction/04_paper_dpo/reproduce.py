"""DPO training pipeline.

Reference: Rafailov et al. 2023 (arXiv:2305.18290)
"""

import torch
import torch.nn as nn


def dpo_loss(pi_logprobs_chosen, pi_logprobs_rejected, ref_logprobs_chosen, ref_logprobs_rejected, beta: float = 0.1):
    """DPO loss implementation."""
    raise NotImplementedError


def train_dpo(model, ref_model, preference_data, beta: float = 0.1, epochs: int = 3):
    """Full DPO training loop."""
    raise NotImplementedError
