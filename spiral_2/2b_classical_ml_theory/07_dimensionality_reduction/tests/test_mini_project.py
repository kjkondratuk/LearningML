"""
Tests for the PCA + t-SNE mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    pca_reconstruction_quality,
)


class TestReconstructionQuality:
    def test_more_components_less_error(self):
        np.random.seed(42)
        X = np.random.randn(100, 50)
        errors = pca_reconstruction_quality(X, [5, 10, 25, 50])
        err_vals = [errors[k] for k in [5, 10, 25, 50]]
        for i in range(len(err_vals) - 1):
            assert err_vals[i] >= err_vals[i + 1] - 1e-10
