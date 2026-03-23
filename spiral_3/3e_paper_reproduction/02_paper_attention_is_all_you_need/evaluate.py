"""BLEU score evaluation."""

import torch

def compute_bleu(references: list[list[str]], hypotheses: list[list[str]]) -> float:
    """Compute corpus-level BLEU score."""
    raise NotImplementedError

def translate_beam_search(model, src: torch.Tensor, beam_size: int = 4, max_len: int = 100):
    """Beam search decoding."""
    raise NotImplementedError
