"""Tests for Policy Gradient exercises."""
import pytest
import torch
from exercises import (
    compute_returns, entropy_bonus, generalized_advantage_estimation, reinforce_loss,
)

class TestComputeReturns:
    def test_known_values(self):
        returns = compute_returns([1.0, 1.0, 1.0], gamma=0.9)
        expected = torch.tensor([1 + 0.9 + 0.81, 1 + 0.9, 1.0])
        assert torch.allclose(returns, expected, atol=1e-4)

    def test_gamma_zero(self):
        returns = compute_returns([1.0, 2.0, 3.0], gamma=0.0)
        assert torch.allclose(returns, torch.tensor([1.0, 2.0, 3.0]))

class TestGAE:
    def test_lambda_zero_is_td_error(self):
        rewards = torch.tensor([1.0, 2.0])
        values = torch.tensor([0.5, 1.0, 0.0])
        gae = generalized_advantage_estimation(rewards, values, gamma=0.9, gae_lambda=0.0)
        delta_0 = 1.0 + 0.9 * 1.0 - 0.5
        assert abs(gae[0].item() - delta_0) < 1e-5

    def test_lambda_one_is_mc_advantage(self):
        rewards = torch.tensor([1.0, 1.0, 1.0])
        values = torch.tensor([0.5, 0.5, 0.5, 0.0])
        gae = generalized_advantage_estimation(rewards, values, gamma=1.0, gae_lambda=1.0)
        # With gamma=1, lambda=1: A_0 = G_0 - V(0) = 3.0 - 0.5 = 2.5
        assert abs(gae[0].item() - 2.5) < 1e-4

class TestEntropy:
    def test_uniform_max(self):
        probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
        ent = entropy_bonus(probs)
        import math
        assert abs(ent.item() - math.log(4)) < 1e-5

    def test_deterministic_zero(self):
        probs = torch.tensor([1.0, 0.0, 0.0])
        ent = entropy_bonus(probs)
        assert abs(ent.item()) < 1e-5
