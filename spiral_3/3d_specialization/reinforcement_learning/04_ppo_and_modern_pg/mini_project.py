"""Mini-Project: PPO on CartPole/LunarLander.

Reference: Schulman et al. 2017 (arXiv:1707.06347)
"""

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    """Shared backbone with separate actor and critic heads."""
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 64):
        raise NotImplementedError
    def forward(self, obs):
        raise NotImplementedError


def train_ppo(env_name: str = "CartPole-v1", total_timesteps: int = 100000) -> dict:
    """Train PPO agent with all modern tricks."""
    raise NotImplementedError
