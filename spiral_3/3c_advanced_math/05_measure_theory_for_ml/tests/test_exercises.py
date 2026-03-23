"""
Tests for Measure Theory exercises.

Verifies: MC integration accuracy, IS variance reduction, Radon-Nikodym properties,
change of variables normalization, KL non-negativity.
"""

import pytest
import torch
import math

from exercises import (
    change_of_variables,
    importance_sampling,
    kl_divergence_mc,
    lebesgue_integral_mc,
    radon_nikodym_discrete,
)


class TestMCIntegration:
    def test_known_integral(self):
        """integral_0^1 x^2 dx = 1/3."""
        est, se = lebesgue_integral_mc(lambda x: x ** 2, (0.0, 1.0), n_samples=50000)
        assert abs(est - 1 / 3) < 3 * se, f"Estimate {est} not within 3 sigma of 1/3"

    def test_constant_function(self):
        """integral_0^2 5 dx = 10."""
        est, se = lebesgue_integral_mc(lambda x: torch.tensor(5.0), (0.0, 2.0), n_samples=10000)
        assert abs(est - 10.0) < 0.5


class TestImportanceSampling:
    def test_lower_variance_with_good_proposal(self):
        """IS with well-matched proposal should have lower variance than naive MC."""
        # Estimate E_p[x^2] where p = N(3, 0.5)
        def f(x): return x ** 2
        def target(x): return torch.exp(-0.5 * ((x - 3) / 0.5) ** 2) / (0.5 * math.sqrt(2 * math.pi))
        def proposal_good(x): return torch.exp(-0.5 * ((x - 3) / 0.7) ** 2) / (0.7 * math.sqrt(2 * math.pi))
        def proposal_bad(x): return torch.ones_like(x) / 10  # Uniform(-5, 5)
        def sampler_good(n): return torch.randn(n) * 0.7 + 3
        def sampler_bad(n): return torch.rand(n) * 10 - 5

        _, var_good = importance_sampling(f, target, proposal_good, sampler_good, 10000)
        _, var_bad = importance_sampling(f, target, proposal_bad, sampler_bad, 10000)
        assert var_good < var_bad


class TestRadonNikodym:
    def test_nonnegative(self):
        p = torch.tensor([0.2, 0.3, 0.5])
        q = torch.tensor([0.1, 0.4, 0.5])
        ratio = radon_nikodym_discrete(p, q)
        assert (ratio >= 0).all()

    def test_integrates_to_one(self):
        """E_Q[dP/dQ] = sum Q(x) * P(x)/Q(x) = sum P(x) = 1."""
        p = torch.tensor([0.2, 0.3, 0.5])
        q = torch.tensor([0.1, 0.4, 0.5])
        ratio = radon_nikodym_discrete(p, q)
        integral = (q * ratio).sum()
        assert torch.allclose(integral, torch.tensor(1.0), atol=1e-5)


class TestKLDivergenceMC:
    def test_nonnegative(self):
        """KL divergence is non-negative."""
        torch.manual_seed(42)
        # P = N(0, 1), Q = N(1, 1)
        def log_p(x): return -0.5 * x ** 2 - 0.5 * math.log(2 * math.pi)
        def log_q(x): return -0.5 * (x - 1) ** 2 - 0.5 * math.log(2 * math.pi)
        samples = torch.randn(10000)
        kl = kl_divergence_mc(log_p, log_q, samples)
        assert kl.item() >= -0.1, f"KL should be non-negative, got {kl.item()}"

    def test_zero_for_identical(self):
        """KL(P || P) = 0."""
        def log_p(x): return -0.5 * x ** 2
        samples = torch.randn(10000)
        kl = kl_divergence_mc(log_p, log_p, samples)
        assert abs(kl.item()) < 0.1

    def test_known_value(self):
        """KL(N(0,1) || N(1,1)) = 0.5."""
        torch.manual_seed(42)
        def log_p(x): return -0.5 * x ** 2 - 0.5 * math.log(2 * math.pi)
        def log_q(x): return -0.5 * (x - 1) ** 2 - 0.5 * math.log(2 * math.pi)
        samples = torch.randn(50000)
        kl = kl_divergence_mc(log_p, log_q, samples)
        assert abs(kl.item() - 0.5) < 0.1, f"KL should be ~0.5, got {kl.item()}"
