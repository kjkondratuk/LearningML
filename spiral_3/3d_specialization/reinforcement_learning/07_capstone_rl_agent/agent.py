"""PPO Agent for continuous control."""

import torch
import torch.nn as nn

class PPOAgent(nn.Module):
    """PPO agent with continuous action space support."""
    def __init__(self, obs_dim: int, act_dim: int):
        raise NotImplementedError
    def act(self, obs: torch.Tensor):
        raise NotImplementedError
    def evaluate(self, obs, actions):
        raise NotImplementedError
