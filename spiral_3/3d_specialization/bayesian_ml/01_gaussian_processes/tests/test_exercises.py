"""Tests for GP exercises."""
import pytest
import torch
from exercises import (
    gp_posterior, gp_prior_sample, rbf_kernel_with_hyperparams,
)

class TestGPPrior:
    def test_zero_mean(self):
        """GP prior has zero mean."""
        X = torch.linspace(0, 1, 50).unsqueeze(1)
        samples = gp_prior_sample(X, rbf_kernel_with_hyperparams, n_samples=1000)
        mean = samples.mean(dim=0)
        assert mean.abs().max() < 0.2

    def test_variance_matches_kernel(self):
        X = torch.linspace(0, 1, 20).unsqueeze(1)
        samples = gp_prior_sample(X, rbf_kernel_with_hyperparams, n_samples=5000)
        var = samples.var(dim=0)
        # RBF kernel with default params: k(x,x) = 1
        assert torch.allclose(var, torch.ones_like(var), atol=0.2)

class TestGPPosterior:
    def test_passes_through_training(self):
        """Posterior mean should pass through training points (noiseless)."""
        X_train = torch.tensor([[0.0], [0.5], [1.0]])
        y_train = torch.tensor([1.0, -1.0, 0.5])
        mean, var = gp_posterior(X_train, y_train, X_train, rbf_kernel_with_hyperparams, noise_var=1e-6)
        assert torch.allclose(mean, y_train, atol=0.01)

    def test_variance_increases_away(self):
        """Posterior variance should increase away from training points."""
        X_train = torch.tensor([[0.5]])
        y_train = torch.tensor([1.0])
        X_near = torch.tensor([[0.6]])
        X_far = torch.tensor([[5.0]])
        _, var_near = gp_posterior(X_train, y_train, X_near, rbf_kernel_with_hyperparams)
        _, var_far = gp_posterior(X_train, y_train, X_far, rbf_kernel_with_hyperparams)
        assert var_far > var_near
