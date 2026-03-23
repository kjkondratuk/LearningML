"""
Tests for Module 03: Probability & Statistics

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    estimate_probability,
    bayes_theorem,
    bayes_disease_test,
    gaussian_pdf,
    sample_and_estimate,
    naive_bayes_classify,
)


# ---------------------------------------------------------------------------
# estimate_probability
# ---------------------------------------------------------------------------

class TestEstimateProbability:
    def test_fair_coin(self):
        outcomes = np.array([0, 1, 0, 1, 0, 1])
        assert estimate_probability(outcomes, 1) == pytest.approx(0.5)

    def test_all_same(self):
        outcomes = np.array([3, 3, 3, 3])
        assert estimate_probability(outcomes, 3) == pytest.approx(1.0)
        assert estimate_probability(outcomes, 0) == pytest.approx(0.0)

    def test_die_roll(self):
        np.random.seed(42)
        outcomes = np.random.randint(1, 7, size=100000)
        p = estimate_probability(outcomes, 6)
        assert p == pytest.approx(1 / 6, abs=0.01)


# ---------------------------------------------------------------------------
# bayes_theorem
# ---------------------------------------------------------------------------

class TestBayesTheorem:
    def test_known_example(self):
        # P(A|B) = 0.8 * 0.1 / 0.2 = 0.4
        result = bayes_theorem(p_a=0.1, p_b_given_a=0.8, p_b=0.2)
        assert result == pytest.approx(0.4)

    def test_certain_prior(self):
        result = bayes_theorem(p_a=1.0, p_b_given_a=0.5, p_b=0.5)
        assert result == pytest.approx(1.0)

    def test_zero_prior(self):
        result = bayes_theorem(p_a=0.0, p_b_given_a=0.9, p_b=0.5)
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# bayes_disease_test
# ---------------------------------------------------------------------------

class TestBayesDiseaseTest:
    def test_classic_problem(self):
        """1% prevalence, 99% sensitivity, 99% specificity -> ~50% PPV."""
        result = bayes_disease_test(
            prevalence=0.01, sensitivity=0.99, specificity=0.99
        )
        assert result == pytest.approx(0.5, abs=0.01)

    def test_high_prevalence(self):
        """50% prevalence, 99% sensitivity, 99% specificity -> ~99% PPV."""
        result = bayes_disease_test(
            prevalence=0.5, sensitivity=0.99, specificity=0.99
        )
        assert result == pytest.approx(0.99, abs=0.01)

    def test_perfect_test(self):
        """Perfect test (100% sensitivity, 100% specificity) -> prevalence."""
        result = bayes_disease_test(
            prevalence=0.05, sensitivity=1.0, specificity=1.0
        )
        assert result == pytest.approx(1.0)

    def test_wrong_way_intuition(self):
        """The common mistake: confusing sensitivity with PPV."""
        result = bayes_disease_test(
            prevalence=0.001, sensitivity=0.99, specificity=0.99
        )
        # Intuition says ~99%, Bayes says ~9%
        assert result < 0.15, (
            "P(disease|positive) should be much lower than sensitivity"
        )


# ---------------------------------------------------------------------------
# gaussian_pdf
# ---------------------------------------------------------------------------

class TestGaussianPdf:
    def test_peak_at_mean(self):
        """PDF is maximized at x = mu."""
        x = np.array([0.0])
        peak = gaussian_pdf(x, mu=0.0, sigma=1.0)
        off_peak = gaussian_pdf(np.array([1.0]), mu=0.0, sigma=1.0)
        assert peak[0] > off_peak[0]

    def test_standard_normal_peak(self):
        """Peak of N(0,1) is 1/sqrt(2*pi) ~ 0.3989."""
        result = gaussian_pdf(np.array([0.0]), mu=0.0, sigma=1.0)
        expected = 1.0 / np.sqrt(2 * np.pi)
        assert result[0] == pytest.approx(expected, abs=1e-10)

    def test_symmetric(self):
        x_pos = gaussian_pdf(np.array([1.0]), mu=0.0, sigma=1.0)
        x_neg = gaussian_pdf(np.array([-1.0]), mu=0.0, sigma=1.0)
        assert x_pos[0] == pytest.approx(x_neg[0])

    def test_narrower_is_taller(self):
        """Smaller sigma -> taller peak (same area under curve)."""
        narrow = gaussian_pdf(np.array([0.0]), mu=0.0, sigma=0.5)
        wide = gaussian_pdf(np.array([0.0]), mu=0.0, sigma=2.0)
        assert narrow[0] > wide[0]

    def test_array_input(self):
        x = np.linspace(-3, 3, 100)
        result = gaussian_pdf(x, mu=0.0, sigma=1.0)
        assert result.shape == (100,)
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# sample_and_estimate
# ---------------------------------------------------------------------------

class TestSampleAndEstimate:
    def test_large_sample_accuracy(self):
        result = sample_and_estimate(mu=5.0, sigma=2.0, n_samples=100000)
        assert result["mean_error"] < 0.05
        assert result["std_error"] < 0.05

    def test_returns_expected_keys(self):
        result = sample_and_estimate(mu=0.0, sigma=1.0, n_samples=100)
        assert "estimated_mean" in result
        assert "estimated_std" in result
        assert "mean_error" in result
        assert "std_error" in result

    def test_reproducible(self):
        r1 = sample_and_estimate(mu=0.0, sigma=1.0, n_samples=50, seed=42)
        r2 = sample_and_estimate(mu=0.0, sigma=1.0, n_samples=50, seed=42)
        assert r1["estimated_mean"] == r2["estimated_mean"]


# ---------------------------------------------------------------------------
# naive_bayes_classify
# ---------------------------------------------------------------------------

class TestNaiveBayesClassify:
    def test_separable_data(self):
        """Two well-separated Gaussian clusters."""
        np.random.seed(42)
        X_class0 = np.random.randn(50, 2) + np.array([0, 0])
        X_class1 = np.random.randn(50, 2) + np.array([5, 5])
        X_train = np.vstack([X_class0, X_class1])
        y_train = np.array([0] * 50 + [1] * 50)

        X_test = np.array([[0, 0], [5, 5], [-1, -1], [6, 6]])
        preds = naive_bayes_classify(X_train, y_train, X_test)
        expected = np.array([0, 1, 0, 1])
        np.testing.assert_array_equal(preds, expected)

    def test_output_shape(self):
        X_train = np.random.randn(20, 3)
        y_train = np.random.randint(0, 2, 20)
        X_test = np.random.randn(5, 3)
        preds = naive_bayes_classify(X_train, y_train, X_test)
        assert preds.shape == (5,)

    def test_three_classes(self):
        np.random.seed(0)
        X0 = np.random.randn(30, 2) + [0, 0]
        X1 = np.random.randn(30, 2) + [5, 0]
        X2 = np.random.randn(30, 2) + [0, 5]
        X_train = np.vstack([X0, X1, X2])
        y_train = np.array([0] * 30 + [1] * 30 + [2] * 30)

        X_test = np.array([[0, 0], [5, 0], [0, 5]])
        preds = naive_bayes_classify(X_train, y_train, X_test)
        np.testing.assert_array_equal(preds, [0, 1, 2])
