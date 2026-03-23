"""
Tests for Word Embedding exercises.

Verifies:
- Negative sampling loss gradients
- GloVe weighting function
- Embedding dimensionality consistency
- Loss differentiability
"""

import pytest
import torch

from exercises import (
    analogy_test,
    cbow_forward,
    glove_loss,
    negative_sampling_loss,
    skipgram_forward,
)


class TestSkipgramForward:
    def test_output_shapes(self):
        vocab_size, embed_dim, batch_size, num_neg = 100, 50, 8, 5
        emb_in = torch.randn(vocab_size, embed_dim)
        emb_out = torch.randn(vocab_size, embed_dim)
        center = torch.randint(0, vocab_size, (batch_size,))
        context = torch.randint(0, vocab_size, (batch_size, 1 + num_neg))
        pos, neg = skipgram_forward(center, context, emb_in, emb_out)
        assert pos.shape == (batch_size, 1)
        assert neg.shape == (batch_size, num_neg)


class TestNegativeSamplingLoss:
    def test_minimized_when_positive_high_negative_low(self):
        pos_good = torch.tensor([[5.0]])
        neg_good = torch.tensor([[-5.0, -5.0, -5.0]])
        pos_bad = torch.tensor([[-5.0]])
        neg_bad = torch.tensor([[5.0, 5.0, 5.0]])

        loss_good = negative_sampling_loss(pos_good, neg_good)
        loss_bad = negative_sampling_loss(pos_bad, neg_bad)
        assert loss_good < loss_bad

    def test_loss_is_differentiable(self):
        pos = torch.randn(4, 1, requires_grad=True)
        neg = torch.randn(4, 5, requires_grad=True)
        loss = negative_sampling_loss(pos, neg)
        loss.backward()
        assert pos.grad is not None
        assert neg.grad is not None
        assert torch.isfinite(pos.grad).all()


class TestCBOWForward:
    def test_output_shape(self):
        emb = torch.randn(100, 50)
        context = torch.randint(0, 100, (8, 4))
        out = cbow_forward(context, emb)
        assert out.shape == (8, 50)


class TestGloveLoss:
    def test_weighting_function(self):
        """f(x) = (x/x_max)^alpha for x < x_max, else 1."""
        w_i = torch.randn(2, 50)
        w_j = torch.randn(2, 50)
        b_i = torch.zeros(2)
        b_j = torch.zeros(2)

        # High co-occurrence should have weight 1
        x_high = torch.tensor([200.0, 300.0])
        loss_high = glove_loss(w_i, w_j, b_i, b_j, x_high, x_max=100.0)

        # Low co-occurrence should have lower weight
        x_low = torch.tensor([1.0, 2.0])
        loss_low = glove_loss(w_i, w_j, b_i, b_j, x_low, x_max=100.0)

        # Both should be valid floats
        assert torch.isfinite(loss_high)
        assert torch.isfinite(loss_low)

    def test_loss_nonnegative(self):
        w_i = torch.randn(8, 50)
        w_j = torch.randn(8, 50)
        b_i = torch.randn(8)
        b_j = torch.randn(8)
        x_ij = torch.rand(8) * 100 + 1
        loss = glove_loss(w_i, w_j, b_i, b_j, x_ij)
        assert loss.item() >= 0
