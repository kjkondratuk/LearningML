"""Tests for PPO exercises."""
import pytest
import torch
from exercises import (
    compute_ratio, ppo_clip_loss,
)

class TestPPOClip:
    def test_positive_advantage_clips_above(self):
        ratio = torch.tensor([2.0])
        adv = torch.tensor([1.0])
        loss = ppo_clip_loss(ratio, adv, epsilon=0.2)
        expected = -(1.2 * 1.0)
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)

    def test_ratio_one_when_same_policy(self):
        log_p = torch.tensor([-1.0, -2.0])
        ratio = compute_ratio(log_p, log_p)
        assert torch.allclose(ratio, torch.ones_like(ratio))
