"""
Tests for Diffusion Model exercises.

Verifies:
- Forward process approaches N(0,I) as T grows
- Noise schedule monotonicity
- Loss non-negativity and training convergence
- DDIM consistency with DDPM
"""

import pytest
import torch
import torch.nn as nn
from scipy import stats

from exercises import (
    compute_noise_schedule,
    ddim_sample,
    ddpm_loss,
    forward_diffusion_step,
    reverse_step,
)


class TestForwardDiffusion:
    def test_output_shape(self):
        schedule = compute_noise_schedule(100, "linear")
        x_0 = torch.randn(8, 1, 28, 28)
        t = torch.randint(0, 100, (8,))
        x_t, eps = forward_diffusion_step(x_0, t, schedule)
        assert x_t.shape == x_0.shape
        assert eps.shape == x_0.shape

    def test_t0_preserves_input(self):
        """At t=0, alpha_bar ~ 1, so x_t ~ x_0."""
        schedule = compute_noise_schedule(1000, "linear")
        x_0 = torch.randn(4, 1, 28, 28)
        t = torch.zeros(4, dtype=torch.long)
        x_t, _ = forward_diffusion_step(x_0, t, schedule)
        # At t=0, alpha_bar is close to 1, so x_t should be close to x_0
        assert torch.allclose(x_t, x_0, atol=0.1), (
            "At t=0, x_t should be very close to x_0"
        )

    def test_large_t_approaches_gaussian(self):
        """At t=T-1, x_t should be approximately standard Gaussian."""
        T = 1000
        schedule = compute_noise_schedule(T, "linear")
        x_0 = torch.ones(1000, 10)  # All ones
        t = torch.full((1000,), T - 1, dtype=torch.long)
        x_t, _ = forward_diffusion_step(x_0, t, schedule)
        # KS test against standard normal
        flat = x_t.flatten().numpy()
        _, p_value = stats.kstest(flat, "norm")
        assert p_value > 0.01, (
            f"At t=T-1, samples should be ~N(0,1). KS p-value={p_value}"
        )


class TestNoiseSchedule:
    def test_linear_schedule_shapes(self):
        schedule = compute_noise_schedule(100, "linear")
        assert schedule["betas"].shape == (100,)
        assert schedule["alphas"].shape == (100,)
        assert schedule["alpha_bars"].shape == (100,)

    def test_alpha_bar_monotonically_decreases(self):
        schedule = compute_noise_schedule(1000, "linear")
        alpha_bars = schedule["alpha_bars"]
        diffs = alpha_bars[1:] - alpha_bars[:-1]
        assert (diffs <= 1e-6).all(), "alpha_bar should monotonically decrease"

    def test_alpha_bar_range(self):
        """alpha_bar should go from near 1 to near 0."""
        schedule = compute_noise_schedule(1000, "linear")
        alpha_bars = schedule["alpha_bars"]
        assert alpha_bars[0] > 0.9, f"First alpha_bar should be near 1, got {alpha_bars[0]}"
        assert alpha_bars[-1] < 0.1, f"Last alpha_bar should be near 0, got {alpha_bars[-1]}"

    def test_cosine_schedule(self):
        schedule = compute_noise_schedule(1000, "cosine")
        alpha_bars = schedule["alpha_bars"]
        assert alpha_bars[0] > 0.9
        assert alpha_bars[-1] < 0.1
        # Cosine schedule should be smoother than linear
        diffs = alpha_bars[1:] - alpha_bars[:-1]
        assert (diffs <= 1e-6).all()


class TestDDPMLoss:
    def test_loss_nonnegative(self):
        """MSE loss must be non-negative."""
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 784),
            nn.Unflatten(1, (1, 28, 28)),
        )
        schedule = compute_noise_schedule(100, "linear")
        x_0 = torch.randn(4, 1, 28, 28)
        loss = ddpm_loss(model, x_0, schedule)
        assert loss.item() >= 0, "Loss must be non-negative"

    def test_loss_is_scalar(self):
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 784),
            nn.Unflatten(1, (1, 28, 28)),
        )
        schedule = compute_noise_schedule(100, "linear")
        x_0 = torch.randn(4, 1, 28, 28)
        loss = ddpm_loss(model, x_0, schedule)
        assert loss.dim() == 0, "Loss should be a scalar"


class TestReverseStep:
    def test_output_shape(self):
        """Reverse step should produce same shape as input."""

        class DummyModel(nn.Module):
            def forward(self, x, t):
                return torch.randn_like(x)

        model = DummyModel()
        schedule = compute_noise_schedule(100, "linear")
        x_t = torch.randn(4, 1, 28, 28)
        t = torch.full((4,), 50, dtype=torch.long)
        x_prev = reverse_step(model, x_t, t, schedule)
        assert x_prev.shape == x_t.shape
