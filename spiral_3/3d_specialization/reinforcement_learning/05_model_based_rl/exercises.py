"""Model-Based RL Exercises. Reference: Hafner et al. 2019 (arXiv:1912.01603)."""

import torch
import torch.nn as nn


def train_dynamics_model(states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor) -> nn.Module:
    """Learn transition model s' = f(s, a)."""
    raise NotImplementedError


def model_predictive_control(dynamics_model: nn.Module, current_state: torch.Tensor, horizon: int, n_candidates: int) -> torch.Tensor:
    """Random shooting MPC: sample action sequences, evaluate, pick best."""
    raise NotImplementedError


def dreamer_world_model_loss(reconstruction: torch.Tensor, kl_loss: torch.Tensor, reward_prediction: torch.Tensor) -> torch.Tensor:
    """RSSM world model loss = reconstruction + kl + reward prediction."""
    raise NotImplementedError


def dyna_q_step(Q, model, real_buffer, model_buffer, n_model_steps: int):
    """Dyna-Q: real experience + model-generated experience."""
    raise NotImplementedError
