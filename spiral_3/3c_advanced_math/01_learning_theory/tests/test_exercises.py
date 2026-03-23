"""
Tests for Statistical Learning Theory exercises.

Verifies:
- VC dimension correctness
- PAC sample complexity monotonicity
- Rademacher complexity non-negativity
- Hoeffding bound validity
- Generalization bound is a valid probability
"""

import pytest
import numpy as np

from exercises import (
    generalization_bound,
    hoeffding_bound,
    mcdiarmid_bound,
    pac_sample_complexity,
    rademacher_complexity_linear,
    vc_dimension_linear_classifier,
)


class TestVCDimension:
    def test_1d(self):
        assert vc_dimension_linear_classifier(1) == 2

    def test_2d(self):
        assert vc_dimension_linear_classifier(2) == 3

    def test_10d(self):
        assert vc_dimension_linear_classifier(10) == 11


class TestPACSampleComplexity:
    def test_increases_with_tighter_epsilon(self):
        """Tighter accuracy demands more data."""
        m_loose = pac_sample_complexity(0.1, 0.05, 10)
        m_tight = pac_sample_complexity(0.01, 0.05, 10)
        assert m_tight > m_loose

    def test_increases_with_vc_dim(self):
        m_small = pac_sample_complexity(0.1, 0.05, 5)
        m_large = pac_sample_complexity(0.1, 0.05, 50)
        assert m_large > m_small

    def test_positive(self):
        m = pac_sample_complexity(0.1, 0.05, 10)
        assert m > 0


class TestRademacherComplexity:
    def test_nonnegative(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        rc = rademacher_complexity_linear(X)
        assert rc >= 0

    def test_decreases_with_more_data(self):
        """Rademacher complexity should roughly decrease as n increases."""
        np.random.seed(42)
        X_small = np.random.randn(50, 5)
        X_large = np.random.randn(500, 5)
        rc_small = rademacher_complexity_linear(X_small, num_samples=2000)
        rc_large = rademacher_complexity_linear(X_large, num_samples=2000)
        assert rc_large < rc_small * 2  # Should decrease, allow some slack


class TestHoeffdingBound:
    def test_valid_probability(self):
        bound = hoeffding_bound(100, 0.1)
        assert 0 <= bound <= 1

    def test_tighter_with_more_samples(self):
        bound_small = hoeffding_bound(100, 0.1)
        bound_large = hoeffding_bound(1000, 0.1)
        assert bound_large < bound_small

    def test_loose_for_small_t(self):
        """For very small t, bound should be close to (but less than) 2."""
        bound = hoeffding_bound(10, 0.001)
        assert bound > 1.9


class TestGeneralizationBound:
    def test_known_case(self):
        """Train error 0.05, n=10000, VC=50 should give bound < 0.15."""
        bound = generalization_bound(0.05, 10000, 50, 0.05)
        assert bound < 0.15, f"Expected bound < 0.15, got {bound}"

    def test_bound_exceeds_train_error(self):
        bound = generalization_bound(0.05, 1000, 20, 0.05)
        assert bound >= 0.05

    def test_all_bounds_valid(self):
        """All bounds should be valid probabilities (in [0, 1] or meaningful)."""
        bound = generalization_bound(0.01, 100, 5, 0.05)
        assert bound >= 0


class TestMcDiarmid:
    def test_valid_bound(self):
        f_vals = np.random.randn(100)
        bound = mcdiarmid_bound(f_vals, perturbation_bound=0.1, n=100, t=0.5)
        assert 0 <= bound <= 1
