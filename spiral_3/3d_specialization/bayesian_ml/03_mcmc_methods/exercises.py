"""MCMC Exercises. Reference: Neal 2011 (arXiv:1206.1901)."""

import torch
import numpy as np


def metropolis_hastings_step(current: torch.Tensor, target_log_prob: callable, proposal_fn: callable) -> tuple:
    """MH accept/reject step. Returns (new_state, accepted)."""
    raise NotImplementedError


def gibbs_sampling_step(state: dict, conditional_samplers: dict) -> dict:
    """Full sweep of Gibbs sampling."""
    raise NotImplementedError


def hamiltonian_dynamics(q: torch.Tensor, p: torch.Tensor, grad_U: callable, epsilon: float, L: int) -> tuple:
    """Leapfrog integrator for HMC. Returns (q_new, p_new)."""
    raise NotImplementedError


def hmc_step(current_q: torch.Tensor, target_log_prob: callable, target_grad: callable, epsilon: float, L: int) -> tuple:
    """Full HMC step with MH correction. Returns (new_q, accepted)."""
    raise NotImplementedError


def effective_sample_size(chain: np.ndarray) -> float:
    """ESS accounting for autocorrelation."""
    raise NotImplementedError


def gelman_rubin_diagnostic(chains: list[np.ndarray]) -> float:
    """R-hat convergence diagnostic."""
    raise NotImplementedError
