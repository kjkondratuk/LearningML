"""
Tests for Probability and Statistics exercises.
"""

import numpy as np
import pytest

from exercises import (
    gaussian_pdf,
    multivariate_gaussian_pdf,
    mle_gaussian,
    map_gaussian,
    bayesian_update,
    conjugate_prior_beta_binomial,
    sample_from_distribution,
    central_limit_theorem_demo,
)


# ── gaussian_pdf ──────────────────────────────────────────────────────────────

class TestGaussianPDF:
    def test_standard_normal_at_zero(self):
        """N(0,1) at x=0 is 1/sqrt(2*pi) ~ 0.3989."""
        val = gaussian_pdf(np.array([0.0]), mu=0.0, sigma=1.0)
        np.testing.assert_allclose(val, [1.0 / np.sqrt(2 * np.pi)], atol=1e-10)

    def test_integrates_to_one(self):
        """Numerical integration of the PDF should be ~1.0."""
        x = np.linspace(-10, 10, 10000)
        pdf = gaussian_pdf(x, mu=0.0, sigma=1.0)
        integral = np.trapz(pdf, x)
        np.testing.assert_allclose(integral, 1.0, atol=1e-3)

    def test_symmetry(self):
        pdf_pos = gaussian_pdf(np.array([2.0]), mu=0.0, sigma=1.0)
        pdf_neg = gaussian_pdf(np.array([-2.0]), mu=0.0, sigma=1.0)
        np.testing.assert_allclose(pdf_pos, pdf_neg, atol=1e-10)

    def test_shifted_mean(self):
        """PDF at the mean should be the maximum."""
        x = np.linspace(-5, 15, 1000)
        pdf = gaussian_pdf(x, mu=5.0, sigma=2.0)
        max_idx = np.argmax(pdf)
        np.testing.assert_allclose(x[max_idx], 5.0, atol=0.05)


# ── multivariate_gaussian_pdf ─────────────────────────────────────────────────

class TestMultivariateGaussianPDF:
    def test_reduces_to_univariate(self):
        """1D multivariate Gaussian should match univariate."""
        x = np.array([1.0])
        mu = np.array([0.0])
        cov = np.array([[1.0]])
        mv_val = multivariate_gaussian_pdf(x, mu, cov)
        uv_val = gaussian_pdf(np.array([1.0]), mu=0.0, sigma=1.0)[0]
        np.testing.assert_allclose(mv_val, uv_val, atol=1e-10)

    def test_max_at_mean(self):
        mu = np.array([1.0, 2.0])
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        val_at_mean = multivariate_gaussian_pdf(mu, mu, cov)
        val_away = multivariate_gaussian_pdf(np.array([3.0, 4.0]), mu, cov)
        assert val_at_mean > val_away


# ── mle_gaussian ──────────────────────────────────────────────────────────────

class TestMLEGaussian:
    def test_matches_numpy(self):
        np.random.seed(42)
        data = np.random.randn(1000) * 3 + 5
        mu, sigma2 = mle_gaussian(data)
        np.testing.assert_allclose(mu, np.mean(data), atol=1e-10)
        np.testing.assert_allclose(sigma2, np.var(data), atol=1e-10)

    def test_simple_data(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mu, sigma2 = mle_gaussian(data)
        np.testing.assert_allclose(mu, 3.0, atol=1e-10)
        np.testing.assert_allclose(sigma2, 2.0, atol=1e-10)


# ── map_gaussian ──────────────────────────────────────────────────────────────

class TestMAPGaussian:
    def test_between_prior_and_mle(self):
        """MAP should be between prior mean and sample mean (shrinkage)."""
        data = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        mu_map = map_gaussian(data, prior_mu=0.0, prior_sigma=1.0, likelihood_sigma=2.0)
        x_bar = np.mean(data)
        assert 0.0 < mu_map < x_bar

    def test_large_data_approaches_mle(self):
        np.random.seed(42)
        data = np.random.randn(10000) + 5.0
        mu_map = map_gaussian(data, prior_mu=0.0, prior_sigma=1.0, likelihood_sigma=1.0)
        np.testing.assert_allclose(mu_map, np.mean(data), atol=0.05)

    def test_tight_prior_approaches_prior_mean(self):
        data = np.array([10.0, 11.0, 12.0])
        mu_map = map_gaussian(data, prior_mu=0.0, prior_sigma=0.01, likelihood_sigma=5.0)
        np.testing.assert_allclose(mu_map, 0.0, atol=0.5)


# ── bayesian_update ───────────────────────────────────────────────────────────

class TestBayesianUpdate:
    def test_uniform_prior(self):
        """Uniform prior + likelihood = normalized likelihood."""
        prior = np.ones(100) / 100
        likelihood = np.random.rand(100)
        posterior = bayesian_update(prior, likelihood, np.linspace(0, 1, 100))
        np.testing.assert_allclose(posterior.sum(), 1.0, atol=1e-10)

    def test_posterior_sums_to_one(self):
        prior = np.array([0.3, 0.5, 0.2])
        likelihood = np.array([0.1, 0.7, 0.2])
        grid = np.array([0, 1, 2])
        posterior = bayesian_update(prior, likelihood, grid)
        np.testing.assert_allclose(posterior.sum(), 1.0, atol=1e-10)


# ── conjugate_prior_beta_binomial ─────────────────────────────────────────────

class TestBetaBinomial:
    def test_update_formula(self):
        a_post, b_post = conjugate_prior_beta_binomial(1, 1, 7, 10)
        assert a_post == 8  # 1 + 7
        assert b_post == 4  # 1 + 3

    def test_no_data(self):
        a_post, b_post = conjugate_prior_beta_binomial(2, 3, 0, 0)
        assert a_post == 2
        assert b_post == 3

    def test_strong_prior(self):
        a_post, b_post = conjugate_prior_beta_binomial(100, 100, 5, 10)
        assert a_post == 105
        assert b_post == 105


# ── sample_from_distribution ──────────────────────────────────────────────────

class TestRejectionSampling:
    def test_samples_from_uniform(self):
        """Sampling from a uniform PDF should give uniform samples."""
        pdf = lambda x: 1.0 if 0 <= x <= 1 else 0.0
        samples = sample_from_distribution(pdf, (0, 1), 1000, seed=42)
        assert len(samples) == 1000
        assert np.all(samples >= 0) and np.all(samples <= 1)

    def test_samples_approximate_gaussian(self):
        """Samples from Gaussian PDF should have approximately correct mean/std."""
        from scipy.stats import norm, kstest
        pdf = lambda x: norm.pdf(x, loc=0, scale=1)
        samples = sample_from_distribution(pdf, (-5, 5), 5000, seed=42)
        # KS test: samples should not be rejected as non-Gaussian
        stat, pvalue = kstest(samples, 'norm', args=(0, 1))
        assert pvalue > 0.01  # not rejected at 1% level


# ── central_limit_theorem_demo ────────────────────────────────────────────────

class TestCLTDemo:
    def test_means_converge_to_gaussian(self):
        """Sample means from an exponential should become Gaussian-like."""
        from scipy.stats import normaltest
        dist = lambda size, rng: rng.exponential(scale=1.0, size=size)
        results = central_limit_theorem_demo(dist, [5, 50, 500], n_experiments=2000)
        # For large sample sizes, the means should pass normality test
        _, pvalue = normaltest(results[500])
        assert pvalue > 0.01
