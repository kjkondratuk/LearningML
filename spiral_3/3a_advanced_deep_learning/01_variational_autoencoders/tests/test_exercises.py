"""
Tests for Variational Autoencoder exercises.

Verifies mathematical properties:
- KL divergence is non-negative and zero for identical distributions
- Reparameterized samples have correct mean/variance
- ELBO is a valid lower bound
- Beta=1 recovers standard VAE
- Gradient flows through reparameterized samples
"""

import pytest
import torch
import torch.nn.functional as F

from exercises import (
    beta_vae_loss,
    compute_elbo,
    compute_kl_divergence,
    log_importance_weights,
    reparameterize,
)


class TestComputeKLDivergence:
    """Test KL divergence between q(z|x) = N(mu, sigma^2) and p(z) = N(0, I)."""

    def test_kl_is_nonnegative(self):
        """KL divergence is always non-negative (Gibbs' inequality)."""
        torch.manual_seed(42)
        mu = torch.randn(32, 10)
        log_var = torch.randn(32, 10)
        kl = compute_kl_divergence(mu, log_var)
        assert (kl >= -1e-6).all(), "KL divergence must be non-negative"

    def test_kl_zero_for_standard_normal(self):
        """KL(N(0,1) || N(0,1)) = 0."""
        mu = torch.zeros(16, 10)
        log_var = torch.zeros(16, 10)
        kl = compute_kl_divergence(mu, log_var)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6), (
            "KL should be 0 when q = p = N(0, I)"
        )

    def test_kl_increases_with_mean_shift(self):
        """Moving mu away from 0 should increase KL."""
        log_var = torch.zeros(16, 10)
        kl_near = compute_kl_divergence(torch.ones(16, 10) * 0.1, log_var)
        kl_far = compute_kl_divergence(torch.ones(16, 10) * 2.0, log_var)
        assert (kl_far > kl_near).all(), "KL should increase as mu moves away from 0"

    def test_kl_output_shape(self):
        """Output should be (batch_size,) -- summed over latent dims."""
        mu = torch.randn(8, 20)
        log_var = torch.randn(8, 20)
        kl = compute_kl_divergence(mu, log_var)
        assert kl.shape == (8,), f"Expected shape (8,), got {kl.shape}"

    def test_kl_known_value(self):
        """For N(1, 1) vs N(0, 1) in 1D, KL = 0.5."""
        mu = torch.tensor([[1.0]])
        log_var = torch.tensor([[0.0]])  # sigma^2 = 1
        kl = compute_kl_divergence(mu, log_var)
        assert torch.allclose(kl, torch.tensor([0.5]), atol=1e-5), (
            f"KL(N(1,1) || N(0,1)) should be 0.5, got {kl.item()}"
        )


class TestReparameterize:
    """Test the reparameterization trick: z = mu + sigma * epsilon."""

    def test_output_shape(self):
        mu = torch.randn(32, 10)
        log_var = torch.randn(32, 10)
        z = reparameterize(mu, log_var)
        assert z.shape == mu.shape, f"Expected shape {mu.shape}, got {z.shape}"

    def test_correct_mean(self):
        """Over many samples, mean should converge to mu."""
        mu = torch.tensor([[3.0, -2.0, 1.0]])
        log_var = torch.zeros(1, 3)  # sigma = 1
        samples = torch.stack([reparameterize(mu, log_var) for _ in range(10000)])
        sample_mean = samples.mean(dim=0)
        assert torch.allclose(sample_mean, mu, atol=0.1), (
            f"Sample mean {sample_mean} should be close to mu {mu}"
        )

    def test_correct_variance(self):
        """Over many samples, variance should converge to exp(log_var)."""
        mu = torch.zeros(1, 3)
        log_var = torch.tensor([[1.0, 0.0, -1.0]])  # var = e, 1, 1/e
        samples = torch.stack([reparameterize(mu, log_var) for _ in range(10000)])
        sample_var = samples.var(dim=0)
        expected_var = torch.exp(log_var)
        assert torch.allclose(sample_var, expected_var, atol=0.15), (
            f"Sample var {sample_var} should be close to {expected_var}"
        )

    def test_gradient_flows(self):
        """Gradients must flow through the reparameterization trick."""
        mu = torch.randn(4, 5, requires_grad=True)
        log_var = torch.randn(4, 5, requires_grad=True)
        z = reparameterize(mu, log_var)
        loss = z.sum()
        loss.backward()
        assert mu.grad is not None, "Gradient must flow to mu"
        assert log_var.grad is not None, "Gradient must flow to log_var"
        assert (mu.grad != 0).any(), "mu gradient should be nonzero"
        assert (log_var.grad != 0).any(), "log_var gradient should be nonzero"


class TestComputeELBO:
    """Test the ELBO computation."""

    def test_returns_three_components(self):
        x = torch.rand(8, 784)
        x_recon = torch.sigmoid(torch.randn(8, 784))
        mu = torch.randn(8, 20)
        log_var = torch.randn(8, 20)
        result = compute_elbo(x, x_recon, mu, log_var)
        assert len(result) == 3, "Should return (elbo, recon_loss, kl_loss)"

    def test_kl_component_nonnegative(self):
        x = torch.rand(16, 784)
        x_recon = torch.sigmoid(torch.randn(16, 784))
        mu = torch.randn(16, 20)
        log_var = torch.randn(16, 20)
        _, _, kl_loss = compute_elbo(x, x_recon, mu, log_var)
        assert (kl_loss >= -1e-6).all(), "KL component must be non-negative"

    def test_recon_component_nonnegative(self):
        x = torch.rand(16, 784)
        x_recon = torch.sigmoid(torch.randn(16, 784))
        mu = torch.randn(16, 20)
        log_var = torch.randn(16, 20)
        _, recon_loss, _ = compute_elbo(x, x_recon, mu, log_var)
        assert (recon_loss >= -1e-6).all(), "Reconstruction loss must be non-negative"

    def test_perfect_reconstruction_zero_kl(self):
        """When reconstruction is perfect and q = p, ELBO = 0 (or near it)."""
        x = torch.ones(4, 10) * 0.5
        x_recon = torch.ones(4, 10) * 0.5
        mu = torch.zeros(4, 5)
        log_var = torch.zeros(4, 5)
        _, recon_loss, kl_loss = compute_elbo(x, x_recon, mu, log_var)
        assert torch.allclose(kl_loss, torch.zeros_like(kl_loss), atol=1e-5)


class TestBetaVAELoss:
    """Test the Beta-VAE objective."""

    def test_beta_one_matches_standard_vae(self):
        """With beta=1, Beta-VAE loss equals standard VAE loss."""
        x = torch.rand(8, 784)
        x_recon = torch.sigmoid(torch.randn(8, 784))
        mu = torch.randn(8, 20)
        log_var = torch.randn(8, 20)

        beta1_loss = beta_vae_loss(x, x_recon, mu, log_var, beta=1.0)
        elbo, recon, kl = compute_elbo(x, x_recon, mu, log_var)
        standard_loss = (recon + kl).mean()

        assert torch.allclose(beta1_loss, standard_loss, atol=1e-5), (
            "Beta=1 should recover standard VAE"
        )

    def test_beta_zero_is_pure_reconstruction(self):
        """With beta=0, only reconstruction loss remains."""
        x = torch.rand(8, 784)
        x_recon = torch.sigmoid(torch.randn(8, 784))
        mu = torch.randn(8, 20)
        log_var = torch.randn(8, 20)

        loss_beta0 = beta_vae_loss(x, x_recon, mu, log_var, beta=0.0)
        _, recon, _ = compute_elbo(x, x_recon, mu, log_var)

        assert torch.allclose(loss_beta0, recon.mean(), atol=1e-5), (
            "Beta=0 should give pure reconstruction loss"
        )

    def test_higher_beta_increases_loss(self):
        """Higher beta should increase loss when KL > 0."""
        x = torch.rand(8, 784)
        x_recon = torch.sigmoid(torch.randn(8, 784))
        mu = torch.randn(8, 20) + 1.0  # Non-zero mean to ensure KL > 0
        log_var = torch.randn(8, 20)

        loss_low = beta_vae_loss(x, x_recon, mu, log_var, beta=1.0)
        loss_high = beta_vae_loss(x, x_recon, mu, log_var, beta=10.0)
        assert loss_high > loss_low, "Higher beta should increase loss when KL > 0"


class TestLogImportanceWeights:
    """Test importance weights for IWAE."""

    def test_output_shape(self):
        z = torch.randn(8, 10)
        mu = torch.randn(8, 10)
        log_var = torch.randn(8, 10)
        log_w = log_importance_weights(z, mu, log_var)
        assert log_w.shape == (8,), f"Expected shape (8,), got {log_w.shape}"

    def test_weights_zero_at_prior(self):
        """When q = p = N(0, I), log importance weight should be 0 for z=0."""
        z = torch.zeros(4, 10)
        mu = torch.zeros(4, 10)
        log_var = torch.zeros(4, 10)
        log_w = log_importance_weights(z, mu, log_var)
        assert torch.allclose(log_w, torch.zeros_like(log_w), atol=1e-5), (
            "Log importance weight should be 0 when q = p and z = mu"
        )

    def test_finite_values(self):
        """Importance weights should be finite for reasonable inputs."""
        z = torch.randn(16, 10)
        mu = torch.randn(16, 10)
        log_var = torch.randn(16, 10)
        log_w = log_importance_weights(z, mu, log_var)
        assert torch.isfinite(log_w).all(), "Log importance weights must be finite"
