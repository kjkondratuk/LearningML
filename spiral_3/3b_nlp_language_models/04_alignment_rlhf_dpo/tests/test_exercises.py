"""
Tests for Alignment exercises.

Verifies:
- Bradley-Terry probability properties
- Reward model loss non-negativity and gradient direction
- PPO clipping behavior
- DPO loss convergence properties
- KL penalty is 0 when policies are identical
"""

import pytest
import torch

from exercises import (
    bradley_terry_probability,
    dpo_loss,
    kl_penalty,
    ppo_clipped_objective,
    reward_model_loss,
)


class TestBradleyTerry:
    def test_equal_rewards_half(self):
        """Equal rewards should give probability 0.5."""
        r = torch.tensor([1.0, 2.0, 3.0])
        prob = bradley_terry_probability(r, r)
        assert torch.allclose(prob, torch.tensor([0.5, 0.5, 0.5]))

    def test_higher_chosen_reward(self):
        """Higher chosen reward should give probability > 0.5."""
        r_chosen = torch.tensor([3.0])
        r_rejected = torch.tensor([1.0])
        prob = bradley_terry_probability(r_chosen, r_rejected)
        assert prob.item() > 0.5

    def test_probability_in_range(self):
        r_c = torch.randn(100)
        r_r = torch.randn(100)
        prob = bradley_terry_probability(r_c, r_r)
        assert (prob >= 0).all() and (prob <= 1).all()


class TestRewardModelLoss:
    def test_nonnegative(self):
        r_c = torch.randn(16)
        r_r = torch.randn(16)
        loss = reward_model_loss(r_c, r_r)
        assert loss.item() >= 0

    def test_gradient_direction(self):
        """Gradient should push r_chosen up and r_rejected down."""
        r_c = torch.tensor([0.0], requires_grad=True)
        r_r = torch.tensor([0.0], requires_grad=True)
        loss = reward_model_loss(r_c, r_r)
        loss.backward()
        assert r_c.grad.item() < 0, "Gradient should decrease loss by increasing r_chosen"
        assert r_r.grad.item() > 0, "Gradient should decrease loss by decreasing r_rejected"


class TestPPOClippedObjective:
    def test_no_clip_when_in_range(self):
        """When ratio is within [1-eps, 1+eps], loss equals ratio * advantage."""
        ratio = torch.tensor([1.0, 1.1, 0.9])
        advantage = torch.tensor([1.0, 1.0, 1.0])
        obj = ppo_clipped_objective(ratio, advantage, epsilon=0.2)
        unclipped = -(ratio * advantage).mean()
        assert torch.allclose(obj, unclipped, atol=1e-5)

    def test_clip_positive_advantage(self):
        """With positive advantage, ratio is clipped above at 1+eps."""
        ratio = torch.tensor([2.0])  # Far above 1+eps
        advantage = torch.tensor([1.0])
        obj = ppo_clipped_objective(ratio, advantage, epsilon=0.2)
        expected = -(1.2 * 1.0)  # Clipped at 1+eps
        assert torch.allclose(obj, torch.tensor(expected), atol=1e-5)

    def test_clip_negative_advantage(self):
        """With negative advantage, ratio is clipped below at 1-eps."""
        ratio = torch.tensor([0.5])  # Far below 1-eps
        advantage = torch.tensor([-1.0])
        obj = ppo_clipped_objective(ratio, advantage, epsilon=0.2)
        expected = -(0.8 * (-1.0))  # Clipped at 1-eps
        assert torch.allclose(obj, torch.tensor(expected), atol=1e-5)


class TestDPOLoss:
    def test_finite_output(self):
        pi_c = torch.randn(8)
        pi_r = torch.randn(8)
        ref_c = torch.randn(8)
        ref_r = torch.randn(8)
        loss = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
        assert torch.isfinite(loss)

    def test_gradient_positive_chosen(self):
        """DPO should increase probability of chosen completions."""
        pi_c = torch.tensor([-2.0], requires_grad=True)
        pi_r = torch.tensor([-2.0], requires_grad=True)
        ref_c = torch.tensor([-2.0])
        ref_r = torch.tensor([-2.0])
        loss = dpo_loss(pi_c, pi_r, ref_c, ref_r, beta=0.1)
        loss.backward()
        # Gradient on pi_chosen should be negative (to increase it and decrease loss)
        assert pi_c.grad.item() < 0, "DPO should increase chosen log prob"
        assert pi_r.grad.item() > 0, "DPO should decrease rejected log prob"


class TestKLPenalty:
    def test_zero_when_identical(self):
        """KL(pi || pi) = 0."""
        logprobs = torch.randn(4, 10)
        kl = kl_penalty(logprobs, logprobs.clone())
        assert torch.allclose(kl, torch.tensor(0.0), atol=1e-6)

    def test_nonnegative(self):
        pi = torch.randn(4, 10)
        ref = torch.randn(4, 10)
        kl = kl_penalty(pi, ref)
        # Note: this is an approximation; may not be strictly non-negative
        # but should be close
        assert kl.item() >= -0.1
