"""
Mini-Project: Train Word2Vec from Scratch

Train Skip-gram with negative sampling on a text corpus.

Reference: Mikolov et al. 2013 (arXiv:1301.3781)
"""

import torch
import torch.nn as nn


class Word2Vec(nn.Module):
    """Skip-gram Word2Vec model."""

    def __init__(self, vocab_size: int, embed_dim: int = 100):
        raise NotImplementedError

    def forward(self, center: torch.Tensor, context: torch.Tensor) -> tuple:
        """Compute positive and negative scores."""
        raise NotImplementedError


def build_vocab(corpus: list[str], min_freq: int = 5) -> dict[str, int]:
    """Build vocabulary from tokenized corpus."""
    raise NotImplementedError


def generate_training_pairs(corpus_ids: list[int], window_size: int = 5, num_negatives: int = 5):
    """Generate (center, context, negatives) training pairs."""
    raise NotImplementedError


def train_word2vec(model: Word2Vec, training_pairs, epochs: int = 5, lr: float = 0.025):
    """Train Word2Vec with negative sampling."""
    raise NotImplementedError
