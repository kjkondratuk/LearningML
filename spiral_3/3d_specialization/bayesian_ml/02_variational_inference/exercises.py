"""Variational Inference Exercises. Reference: Blei et al. 2017 (arXiv:1601.00670)."""

import torch


def mean_field_elbo(log_joint: callable, q_samples: torch.Tensor, q_log_prob: torch.Tensor) -> torch.Tensor:
    """ELBO = E_q[log p(x,z)] - E_q[log q(z)]."""
    raise NotImplementedError


def cavi_update_gaussian(other_params: dict, data: torch.Tensor, prior_params: dict) -> dict:
    """Coordinate ascent VI update for Gaussian mixture."""
    raise NotImplementedError


def stochastic_vi_step(elbo_gradient_estimate: torch.Tensor, params: torch.Tensor, lr: float) -> torch.Tensor:
    """SVI gradient step."""
    raise NotImplementedError


def reparameterized_gradient(mu: torch.Tensor, sigma: torch.Tensor, n_samples: int) -> tuple:
    """Gradient of ELBO via reparameterization."""
    raise NotImplementedError


def black_box_vi_gradient(log_joint: callable, q_sample: torch.Tensor, q_log_prob: torch.Tensor) -> torch.Tensor:
    """Score function estimator (REINFORCE for VI)."""
    raise NotImplementedError
