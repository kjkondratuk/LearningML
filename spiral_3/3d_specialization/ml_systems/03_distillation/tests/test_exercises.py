"""Tests for Knowledge Distillation."""
import pytest
from exercises import (
    distillation_loss, soft_labels, feature_distillation_loss, attention_transfer_loss,
)

class TestDistillation:
    def test_t1_alpha0_is_ce(self):
        """T=1, alpha=0 should reduce to standard cross-entropy."""
        import torch
        s = torch.randn(4, 10)
        t = torch.randn(4, 10)
        labels = torch.randint(0, 10, (4,))
        loss = distillation_loss(s, t, labels, temperature=1.0, alpha=0.0)
        import torch.nn.functional as F
        ce = F.cross_entropy(s, labels)
        assert abs(loss.item() - ce.item()) < 0.01
