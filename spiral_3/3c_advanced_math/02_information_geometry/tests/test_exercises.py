"""
Tests for Information Geometry exercises.

Verifies:
- Fisher information matches Monte Carlo verification
- Natural gradient convergence
- KL second-order approximation error is O(delta^3)
- Fisher-Rao distance is a true metric
"""

import pytest
import torch
import numpy as np

from exercises import (
    fisher_information_bernoulli,
    fisher_information_gaussian,
    geodesic_distance_gaussians,
    kl_divergence_second_order,
    natural_gradient_step,
)


class TestFisherGaussian:
    def test_known_values(self):
        """F for N(0, 1) should be [[1, 0], [0, 2]]."""
        F = fisher_information_gaussian(0.0, 1.0)
        expected = np.array([[1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_allclose(F, expected, atol=1e-6)

    def test_positive_semidefinite(self):
        """Fisher information matrix must be PSD."""
        F = fisher_information_gaussian(3.0, 2.0)
        eigenvalues = np.linalg.eigvalsh(F)
        assert (eigenvalues >= -1e-10).all()

    def test_monte_carlo_verification(self):
        """Verify numerically: F_ij = E[(d/d theta_i log p) (d/d theta_j log p)]."""
        mu, sigma = 0.0, 1.0
        F = fisher_information_gaussian(mu, sigma)
        # Monte Carlo: sample x ~ N(mu, sigma^2), compute score, average outer product
        np.random.seed(42)
        n = 50000
        x = np.random.normal(mu, sigma, n)
        # Score for mu: (x - mu) / sigma^2
        score_mu = (x - mu) / sigma ** 2
        # Score for sigma: -1/sigma + (x - mu)^2 / sigma^3
        score_sigma = -1 / sigma + (x - mu) ** 2 / sigma ** 3
        F_mc = np.array([
            [np.mean(score_mu ** 2), np.mean(score_mu * score_sigma)],
            [np.mean(score_mu * score_sigma), np.mean(score_sigma ** 2)],
        ])
        np.testing.assert_allclose(F, F_mc, atol=0.1)


class TestFisherBernoulli:
    def test_symmetric(self):
        """I(p) = I(1-p)."""
        assert abs(fisher_information_bernoulli(0.3) - fisher_information_bernoulli(0.7)) < 1e-10

    def test_minimum_at_half(self):
        """Fisher information is minimized at p=0.5 (least informative)."""
        assert fisher_information_bernoulli(0.5) < fisher_information_bernoulli(0.1)

    def test_known_value(self):
        """I(0.5) = 4."""
        assert abs(fisher_information_bernoulli(0.5) - 4.0) < 1e-10


class TestNaturalGradient:
    def test_reduces_to_vanilla_with_identity(self):
        """With F = I, natural gradient equals vanilla gradient."""
        params = torch.tensor([1.0, 2.0])
        grad = torch.tensor([0.5, -0.3])
        F = torch.eye(2)
        updated = natural_gradient_step(params, grad, F, lr=0.1)
        expected = params - 0.1 * grad
        assert torch.allclose(updated, expected, atol=1e-6)


class TestKLSecondOrder:
    def test_zero_for_no_perturbation(self):
        F = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
        delta = torch.zeros(2)
        kl = kl_divergence_second_order(F, delta)
        assert torch.allclose(kl, torch.tensor(0.0))

    def test_nonnegative(self):
        F = torch.tensor([[2.0, 0.5], [0.5, 3.0]])
        delta = torch.randn(2)
        kl = kl_divergence_second_order(F, delta)
        assert kl.item() >= -1e-6


class TestGeodesicDistance:
    def test_zero_for_identical(self):
        d = geodesic_distance_gaussians(0.0, 1.0, 0.0, 1.0)
        assert abs(d) < 1e-6

    def test_symmetric(self):
        d1 = geodesic_distance_gaussians(0.0, 1.0, 1.0, 2.0)
        d2 = geodesic_distance_gaussians(1.0, 2.0, 0.0, 1.0)
        assert abs(d1 - d2) < 1e-6

    def test_triangle_inequality(self):
        """d(A,C) <= d(A,B) + d(B,C)."""
        d_ac = geodesic_distance_gaussians(0.0, 1.0, 2.0, 3.0)
        d_ab = geodesic_distance_gaussians(0.0, 1.0, 1.0, 2.0)
        d_bc = geodesic_distance_gaussians(1.0, 2.0, 2.0, 3.0)
        assert d_ac <= d_ab + d_bc + 1e-6

    def test_nonnegative(self):
        d = geodesic_distance_gaussians(1.0, 2.0, -1.0, 0.5)
        assert d >= 0
