"""Tests for PPO mini-project."""
import pytest
import torch
from mini_project import ActorCritic

class TestActorCritic:
    def test_output_shapes(self):
        model = ActorCritic(obs_dim=4, act_dim=2)
        obs = torch.randn(8, 4)
        out = model(obs)
        assert len(out) >= 2  # Should return policy and value
