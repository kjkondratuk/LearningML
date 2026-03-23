"""Tests for Transformer reproduction."""
import pytest
import torch
from reproduce import (
    Transformer, label_smoothing_loss, transformer_lr_schedule,
)

class TestTransformer:
    def test_dimensions_match_paper(self):
        model = Transformer(src_vocab=1000, tgt_vocab=1000, d_model=512, n_heads=8, n_layers=6)
        src = torch.randint(0, 1000, (2, 20))
        tgt = torch.randint(0, 1000, (2, 15))
        out = model(src, tgt)
        assert out.shape == (2, 15, 1000)

class TestLabelSmoothing:
    def test_epsilon_zero_is_standard_ce(self):
        logits = torch.randn(4, 10)
        target = torch.randint(0, 10, (4,))
        ls_loss = label_smoothing_loss(logits, target, epsilon=0.0)
        ce_loss = torch.nn.functional.cross_entropy(logits, target)
        assert abs(ls_loss.item() - ce_loss.item()) < 0.01

class TestLRSchedule:
    def test_warmup_increases(self):
        lr1 = transformer_lr_schedule(512, step=100, warmup_steps=4000)
        lr2 = transformer_lr_schedule(512, step=200, warmup_steps=4000)
        assert lr2 > lr1

    def test_post_warmup_decreases(self):
        lr1 = transformer_lr_schedule(512, step=5000, warmup_steps=4000)
        lr2 = transformer_lr_schedule(512, step=10000, warmup_steps=4000)
        assert lr2 < lr1
