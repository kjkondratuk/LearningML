"""Result analysis with statistical tests."""

import numpy as np


def compute_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple:
    """Compute confidence interval for a set of measurements."""
    raise NotImplementedError


def significance_test(baseline_results: np.ndarray, extension_results: np.ndarray) -> dict:
    """Statistical significance test between baseline and extension."""
    raise NotImplementedError
