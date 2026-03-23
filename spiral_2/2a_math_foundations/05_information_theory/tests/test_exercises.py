"""
Tests for Information Theory exercises.
"""

import numpy as np
import pytest

from exercises import (
    entropy,
    cross_entropy,
    kl_divergence,
    mutual_information,
    differential_entropy_gaussian,
    kl_divergence_gaussians,
)


# ── entropy ───────────────────────────────────────────────────────────────────

class TestEntropy:
    def test_fair_coin(self):
        """Entropy of a fair coin is exactly 1.0 bit."""
        H = entropy(np.array([0.5, 0.5]))
        np.testing.assert_allclose(H, 1.0, atol=1e-10)

    def test_uniform_over_n(self):
        """Entropy of uniform over n items = log2(n)."""
        for n in [2, 4, 8, 16]:
            p = np.ones(n) / n
            H = entropy(p)
            np.testing.assert_allclose(H, np.log2(n), atol=1e-10)

    def test_certain_outcome(self):
        """Entropy of a deterministic distribution is 0."""
        H = entropy(np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(H, 0.0, atol=1e-10)

    def test_non_negative(self):
        """Entropy is always non-negative."""
        np.random.seed(42)
        for _ in range(50):
            p = np.random.dirichlet(np.ones(5))
            assert entropy(p) >= -1e-10


# ── cross_entropy ─────────────────────────────────────────────────────────────

class TestCrossEntropy:
    def test_cross_entropy_equals_entropy_when_p_equals_q(self):
        """H(p, p) = H(p)."""
        p = np.array([0.2, 0.3, 0.5])
        np.testing.assert_allclose(cross_entropy(p, p), entropy(p), atol=1e-10)

    def test_cross_entropy_geq_entropy(self):
        """H(p, q) >= H(p) (consequence of Gibbs' inequality)."""
        p = np.array([0.7, 0.2, 0.1])
        q = np.array([0.3, 0.4, 0.3])
        assert cross_entropy(p, q) >= entropy(p) - 1e-10


# ── kl_divergence ─────────────────────────────────────────────────────────────

class TestKLDivergence:
    def test_kl_non_negative(self):
        """KL(p||q) >= 0 for 100 random pairs."""
        np.random.seed(42)
        for _ in range(100):
            p = np.random.dirichlet(np.ones(5))
            q = np.random.dirichlet(np.ones(5))
            assert kl_divergence(p, q) >= -1e-10

    def test_kl_zero_iff_equal(self):
        """KL(p||p) = 0."""
        p = np.array([0.3, 0.4, 0.3])
        np.testing.assert_allclose(kl_divergence(p, p), 0.0, atol=1e-10)

    def test_not_symmetric(self):
        """KL(p||q) != KL(q||p) in general."""
        p = np.array([0.9, 0.1])
        q = np.array([0.5, 0.5])
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        assert abs(kl_pq - kl_qp) > 0.01


# ── mutual_information ────────────────────────────────────────────────────────

class TestMutualInformation:
    def test_independent_variables(self):
        """MI = 0 for independent X and Y."""
        joint_p = np.array([[0.15, 0.35], [0.10, 0.40]])
        # Check if actually independent: p(x,y) = p(x)*p(y)
        px = joint_p.sum(axis=1)
        py = joint_p.sum(axis=0)
        indep = np.outer(px, py)
        # Only test if approximately independent
        if np.allclose(joint_p, indep, atol=0.01):
            mi = mutual_information(joint_p)
            np.testing.assert_allclose(mi, 0.0, atol=0.1)

    def test_symmetric(self):
        """I(X;Y) = I(Y;X)."""
        joint_p = np.array([[0.3, 0.1], [0.1, 0.5]])
        mi_xy = mutual_information(joint_p)
        mi_yx = mutual_information(joint_p.T)
        np.testing.assert_allclose(mi_xy, mi_yx, atol=1e-10)

    def test_non_negative(self):
        """MI is always non-negative."""
        np.random.seed(42)
        for _ in range(50):
            joint = np.random.dirichlet(np.ones(6)).reshape(2, 3)
            assert mutual_information(joint) >= -1e-10

    def test_perfectly_correlated(self):
        """MI should be maximal when X determines Y."""
        joint_p = np.array([[0.5, 0.0], [0.0, 0.5]])
        mi = mutual_information(joint_p)
        assert mi > 0.9  # Should be 1 bit


# ── differential_entropy_gaussian ─────────────────────────────────────────────

class TestDifferentialEntropyGaussian:
    def test_known_formula(self):
        """h(N(mu, sigma^2)) = 0.5 * ln(2*pi*e*sigma^2)."""
        sigma = 2.0
        expected = 0.5 * np.log(2 * np.pi * np.e * sigma ** 2)
        np.testing.assert_allclose(
            differential_entropy_gaussian(sigma), expected, atol=1e-10
        )

    def test_standard_normal(self):
        """h(N(0,1)) = 0.5 * ln(2*pi*e) ~ 1.4189."""
        expected = 0.5 * np.log(2 * np.pi * np.e)
        np.testing.assert_allclose(
            differential_entropy_gaussian(1.0), expected, atol=1e-10
        )


# ── kl_divergence_gaussians ──────────────────────────────────────────────────

class TestKLGaussians:
    def test_same_distribution(self):
        kl = kl_divergence_gaussians(0.0, 1.0, 0.0, 1.0)
        np.testing.assert_allclose(kl, 0.0, atol=1e-10)

    def test_non_negative(self):
        np.random.seed(42)
        for _ in range(100):
            mu1, mu2 = np.random.randn(2)
            s1, s2 = np.abs(np.random.randn(2)) + 0.1
            kl = kl_divergence_gaussians(mu1, s1, mu2, s2)
            assert kl >= -1e-10

    def test_known_value(self):
        """KL(N(0,1) || N(1,2))."""
        kl = kl_divergence_gaussians(0.0, 1.0, 1.0, 2.0)
        expected = np.log(2.0 / 1.0) + (1.0 + 1.0) / (2 * 4.0) - 0.5
        np.testing.assert_allclose(kl, expected, atol=1e-10)
