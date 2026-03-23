"""
Tests for Scaling Laws & ICL exercises.

Verifies:
- Power law fit quality on synthetic data
- Chinchilla ratio properties
- FLOPs estimate order of magnitude
- Bayesian ICL posterior concentration
- Emergent ability detection
"""

import pytest
import numpy as np
import torch

from exercises import (
    bayesian_icl_posterior,
    chinchilla_optimal_ratio,
    compute_flops_estimate,
    emergent_ability_metric,
    fit_power_law,
)


class TestFitPowerLaw:
    def test_good_fit_on_synthetic(self):
        """Power law fit should have R^2 > 0.95 on clean power law data."""
        np.random.seed(42)
        C = np.logspace(15, 22, 20)
        L = 5.0 * C ** (-0.07) + 1.5 + np.random.normal(0, 0.01, 20)
        result = fit_power_law(C, L)
        assert result["r_squared"] > 0.95, (
            f"R^2 should be > 0.95, got {result['r_squared']}"
        )

    def test_alpha_positive(self):
        """The exponent alpha should be positive (loss decreases with compute)."""
        C = np.logspace(15, 22, 20)
        L = 3.0 * C ** (-0.05) + 2.0
        result = fit_power_law(C, L)
        assert result["alpha"] > 0


class TestChinchillaRatio:
    def test_params_and_tokens_scale_equally(self):
        """N and D should scale approximately equally with compute."""
        result_small = chinchilla_optimal_ratio(1e18)
        result_large = chinchilla_optimal_ratio(1e21)

        ratio_small = result_small["optimal_params"] / result_small["optimal_tokens"]
        ratio_large = result_large["optimal_params"] / result_large["optimal_tokens"]

        # Ratios should be similar (within an order of magnitude)
        assert abs(np.log10(ratio_small / ratio_large)) < 1.0


class TestFLOPsEstimate:
    def test_gpt3_order_of_magnitude(self):
        """GPT-3: 175B params, 300B tokens -> ~3.15e23 FLOPs."""
        flops = compute_flops_estimate(175e9, 300e9)
        assert 1e23 < flops < 1e24, f"GPT-3 FLOPs should be ~3e23, got {flops}"


class TestBayesianICL:
    def test_posterior_concentrates(self):
        """With more examples, posterior should concentrate (lower entropy)."""
        prior = torch.ones(5) / 5
        true_h = 2

        def likelihood_fn(h, x, y):
            return 0.9 if h == true_h and y == x + 1 else 0.1

        examples_few = [(1, 2)]
        examples_many = [(1, 2), (3, 4), (5, 6), (7, 8)]

        post_few = bayesian_icl_posterior(prior, likelihood_fn, examples_few)
        post_many = bayesian_icl_posterior(prior, likelihood_fn, examples_many)

        entropy_few = -(post_few * post_few.log()).sum()
        entropy_many = -(post_many * post_many.log()).sum()
        assert entropy_many < entropy_few, "More examples should concentrate posterior"

    def test_posterior_sums_to_one(self):
        prior = torch.ones(3) / 3

        def likelihood_fn(h, x, y):
            return 0.5

        post = bayesian_icl_posterior(prior, likelihood_fn, [(1, 1)])
        assert torch.allclose(post.sum(), torch.tensor(1.0), atol=1e-5)


class TestEmergentAbility:
    def test_detects_step_function(self):
        """Should detect step-function behavior as emergent."""
        sizes = np.logspace(7, 11, 20)
        accuracy = np.where(sizes > 1e9, 0.8, 0.1) + np.random.normal(0, 0.02, 20)
        result = emergent_ability_metric(sizes, accuracy)
        assert result["is_emergent"], "Should detect step-function as emergent"

    def test_detects_gradual_improvement(self):
        """Gradual linear improvement should NOT be classified as emergent."""
        sizes = np.logspace(7, 11, 20)
        accuracy = np.log10(sizes) / 11.0 + np.random.normal(0, 0.01, 20)
        result = emergent_ability_metric(sizes, accuracy)
        assert not result["is_emergent"], "Gradual improvement is not emergent"
