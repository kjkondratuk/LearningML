"""Tests for MCMC exercises."""
import pytest
import torch
import numpy as np
from scipy import stats
from exercises import (
    effective_sample_size, gelman_rubin_diagnostic, hamiltonian_dynamics, hmc_step, metropolis_hastings_step,
)

class TestMH:
    def test_samples_gaussian(self):
        """MH on known Gaussian should have correct mean."""
        def log_prob(x):
            return -0.5 * x**2
        def proposal(x):
            return x + 0.5 * torch.randn_like(x)
        current = torch.tensor([0.0])
        samples = []
        for _ in range(5000):
            current, _ = metropolis_hastings_step(current, log_prob, proposal)
            samples.append(current.item())
        samples = np.array(samples[1000:])
        _, p_value = stats.kstest(samples, "norm")
        assert p_value > 0.01

class TestHMC:
    def test_hamiltonian_conserved(self):
        """Leapfrog should approximately conserve the Hamiltonian."""
        q = torch.tensor([1.0])
        p = torch.tensor([0.5])
        def grad_U(q): return q  # U(q) = 0.5*q^2
        q_new, p_new = hamiltonian_dynamics(q, p, grad_U, epsilon=0.01, L=100)
        H_old = 0.5 * q**2 + 0.5 * p**2
        H_new = 0.5 * q_new**2 + 0.5 * p_new**2
        assert abs(H_old.item() - H_new.item()) < 0.1

class TestESS:
    def test_independent_samples(self):
        """Independent samples should have ESS = n."""
        np.random.seed(42)
        chain = np.random.randn(1000)
        ess = effective_sample_size(chain)
        assert ess > 800  # Should be close to 1000

class TestGelmanRubin:
    def test_converged_chains(self):
        np.random.seed(42)
        chains = [np.random.randn(1000) for _ in range(4)]
        r_hat = gelman_rubin_diagnostic(chains)
        assert r_hat < 1.1
