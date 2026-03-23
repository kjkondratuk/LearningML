"""Tests for DPO reproduction."""
import pytest
import torch
from reproduce import dpo_loss

class TestDPOLoss:
    def test_no_nan(self):
        pi_c = torch.randn(8)
        pi_r = torch.randn(8)
        ref_c = torch.randn(8)
        ref_r = torch.randn(8)
        loss = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
        assert torch.isfinite(loss)

    def test_gradient_direction(self):
        """DPO should increase chosen and decrease rejected."""
        pi_c = torch.tensor([-1.0], requires_grad=True)
        pi_r = torch.tensor([-1.0], requires_grad=True)
        ref_c = torch.tensor([-1.0])
        ref_r = torch.tensor([-1.0])
        loss = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
        loss.backward()
        assert pi_c.grad.item() < 0  # Increase chosen
        assert pi_r.grad.item() > 0  # Decrease rejected
