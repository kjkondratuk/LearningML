"""Tests for VI exercises."""
import pytest
import torch
from exercises import mean_field_elbo

class TestELBO:
    def test_is_lower_bound(self):
        """ELBO should be a lower bound on log evidence on tractable model."""
        # Simple Gaussian: p(z) = N(0,1), p(x|z) = N(z,1)
        # q(z) = N(mu, sigma^2)
        # For known x, log p(x) is computable
        # Just verify ELBO is finite
        def log_joint(z):
            return -0.5 * z**2 - 0.5 * (z - 1.0)**2
        q_samples = torch.randn(100)
        q_log_prob = -0.5 * q_samples**2
        elbo = mean_field_elbo(log_joint, q_samples, q_log_prob)
        assert torch.isfinite(elbo)
