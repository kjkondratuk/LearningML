"""
Dataset Preparation for LM Fine-tuning

Prepare instruction-following datasets in standard formats.
"""

import torch


def load_alpaca_format(path: str) -> list[dict]:
    """Load dataset in Alpaca format (instruction, input, output)."""
    raise NotImplementedError


def tokenize_for_sft(examples: list[dict], tokenizer, max_length: int = 512) -> dict:
    """Tokenize examples for supervised fine-tuning."""
    raise NotImplementedError


def prepare_preference_pairs(
    sft_examples: list[dict], model, tokenizer
) -> list[dict]:
    """Generate preference pairs for DPO training."""
    raise NotImplementedError
