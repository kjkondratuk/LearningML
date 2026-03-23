"""Tests for Word2Vec mini-project."""

import pytest
import torch

from mini_project import Word2Vec


class TestWord2Vec:
    def test_embedding_shapes(self):
        model = Word2Vec(vocab_size=1000, embed_dim=100)
        center = torch.randint(0, 1000, (4,))
        context = torch.randint(0, 1000, (4, 6))
        pos, neg = model(center, context)
        assert pos.shape == (4, 1)
        assert neg.shape == (4, 5)
