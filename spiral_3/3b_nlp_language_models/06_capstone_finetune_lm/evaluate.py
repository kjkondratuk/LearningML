"""
Evaluation for Fine-tuned Language Model

Compute perplexity, downstream task metrics, and qualitative analysis.
"""

import torch
import torch.nn as nn


def compute_perplexity(model: nn.Module, eval_loader, device: str = "cuda") -> float:
    """Compute perplexity on evaluation dataset."""
    raise NotImplementedError


def generate_samples(model: nn.Module, tokenizer, prompts: list[str], max_tokens: int = 256):
    """Generate completions for evaluation prompts."""
    raise NotImplementedError


def compare_models(base_model, sft_model, dpo_model, prompts: list[str], tokenizer):
    """Side-by-side comparison of base, SFT, and DPO model outputs."""
    raise NotImplementedError
