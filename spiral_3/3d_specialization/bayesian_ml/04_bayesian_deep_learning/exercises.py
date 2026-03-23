"""Bayesian Deep Learning Exercises."""

import torch
import torch.nn as nn


def mc_dropout_predict(model: nn.Module, x: torch.Tensor, n_forward_passes: int = 50, dropout_rate: float = 0.1) -> tuple:
    """Predictive mean and variance via MC Dropout. Returns (mean, variance)."""
    raise NotImplementedError


def bayes_by_backprop_loss(log_likelihood: torch.Tensor, kl_weight: float, weight_prior, weight_posterior) -> torch.Tensor:
    """ELBO for Bayes by Backprop: log_lik - kl_weight * KL(posterior || prior)."""
    raise NotImplementedError


def deep_ensemble_predict(models: list[nn.Module], x: torch.Tensor) -> tuple:
    """Mean and variance from ensemble. Returns (mean, variance)."""
    raise NotImplementedError


def calibration_error(predicted_probs: torch.Tensor, true_labels: torch.Tensor, n_bins: int = 15) -> float:
    """Expected Calibration Error."""
    raise NotImplementedError
