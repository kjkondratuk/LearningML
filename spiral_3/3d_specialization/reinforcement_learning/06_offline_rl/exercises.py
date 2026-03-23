"""Offline RL Exercises. Reference: Kumar et al. 2020 (arXiv:2006.04779)."""

import torch


def cql_loss(Q_values: torch.Tensor, dataset_actions: torch.Tensor, all_actions: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """CQL loss: standard Q loss + alpha * (logsumexp(Q) - E_data[Q])."""
    raise NotImplementedError


def iql_value_loss(Q: torch.Tensor, V: torch.Tensor, tau: float = 0.7) -> torch.Tensor:
    """IQL expectile regression loss."""
    raise NotImplementedError


def distributional_shift_metric(behavior_policy: torch.Tensor, learned_policy: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    """KL divergence between behavior and learned policies."""
    raise NotImplementedError


def pessimistic_value_estimate(Q_ensemble: torch.Tensor, actions: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """Mean - beta * std across ensemble members."""
    raise NotImplementedError
