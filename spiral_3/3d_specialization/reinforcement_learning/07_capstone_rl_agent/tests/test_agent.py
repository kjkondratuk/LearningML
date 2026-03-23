"""Tests for RL agent."""
import pytest
import torch
from agent import PPOAgent

class TestPPOAgent:
    def test_act_shape(self):
        agent = PPOAgent(obs_dim=11, act_dim=3)
        obs = torch.randn(1, 11)
        action = agent.act(obs)
        assert action.shape[-1] == 3
