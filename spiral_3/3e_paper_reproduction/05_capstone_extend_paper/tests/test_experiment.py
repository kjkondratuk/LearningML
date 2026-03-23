"""Tests for extension experiment."""
import pytest
from experiment import run_baseline, run_extension
from analyze import compute_confidence_interval
import numpy as np

class TestExperiment:
    def test_baseline_runs(self):
        """Baseline should execute without error."""
        pytest.skip("Requires full implementation")

    def test_extension_runs(self):
        """Extension should execute without error."""
        pytest.skip("Requires full implementation")

class TestAnalysis:
    def test_confidence_interval(self):
        vals = np.random.randn(100)
        low, high = compute_confidence_interval(vals)
        assert low < high
        assert low < np.mean(vals) < high
