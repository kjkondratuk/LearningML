"""
Tests for the Bayesian Coin-Flip Inference mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    beta_pdf,
    sequential_bayesian_update,
    posterior_mean,
    posterior_mode,
    mle_estimate,
    strong_prior_resistance,
)


class TestBetaPDF:
    def test_uniform_prior(self):
        """Beta(1,1) is the uniform distribution."""
        x = np.linspace(0.01, 0.99, 100)
        pdf = beta_pdf(x, 1.0, 1.0)
        np.testing.assert_allclose(pdf, np.ones_like(pdf), atol=1e-10)

    def test_integrates_to_one(self):
        x = np.linspace(0.001, 0.999, 10000)
        pdf = beta_pdf(x, 2.0, 5.0)
        integral = np.trapz(pdf, x)
        np.testing.assert_allclose(integral, 1.0, atol=1e-2)


class TestSequentialUpdate:
    def test_all_heads(self):
        flips = [1, 1, 1, 1, 1]
        posteriors = sequential_bayesian_update(flips, 1.0, 1.0)
        assert len(posteriors) == 5
        a_final, b_final = posteriors[-1]
        assert a_final == 6  # 1 + 5
        assert b_final == 1  # 1 + 0

    def test_all_tails(self):
        flips = [0, 0, 0]
        posteriors = sequential_bayesian_update(flips, 1.0, 1.0)
        a_final, b_final = posteriors[-1]
        assert a_final == 1
        assert b_final == 4


class TestPosteriorStatistics:
    def test_mean(self):
        np.testing.assert_allclose(posterior_mean(3.0, 7.0), 0.3, atol=1e-10)

    def test_mode(self):
        np.testing.assert_allclose(posterior_mode(3.0, 7.0), 2.0 / 8.0, atol=1e-10)


class TestMLEEstimate:
    def test_fair_coin(self):
        flips = [0, 1, 0, 1, 0, 1]
        assert mle_estimate(flips) == 0.5

    def test_all_heads(self):
        flips = [1, 1, 1, 1]
        assert mle_estimate(flips) == 1.0


class TestStrongPriorResistance:
    def test_weak_prior_moves_toward_true(self):
        result = strong_prior_resistance(true_p=0.8, n_flips=100, seed=42)
        a, b = result['weak_posteriors'][-1]
        weak_mean = a / (a + b)
        assert abs(weak_mean - 0.8) < 0.15

    def test_strong_prior_resists(self):
        result = strong_prior_resistance(true_p=0.8, n_flips=100, seed=42)
        a, b = result['strong_posteriors'][-1]
        strong_mean = a / (a + b)
        # Strong Beta(100,100) prior centered at 0.5 should resist
        assert abs(strong_mean - 0.5) < abs(strong_mean - 0.8)
