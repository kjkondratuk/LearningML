"""Tests for Model-Based RL exercises."""
import pytest
import torch
from exercises import (
    dreamer_world_model_loss,
)

class TestDreamerLoss:
    def test_nonnegative(self):
        recon = torch.tensor(1.0)
        kl = torch.tensor(0.5)
        reward = torch.tensor(0.3)
        loss = dreamer_world_model_loss(recon, kl, reward)
        assert loss.item() >= 0
