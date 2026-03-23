"""
Tests for the Regularization Ablation Study mini-project.
"""

import numpy as np
import pytest

from mini_project import (
    create_overfit_dataset,
)


class TestCreateDataset:
    def test_shapes(self):
        data = create_overfit_dataset(n_train=100, n_test=500, d=50)
        X_train, y_train, X_test, y_test = data
        assert X_train.shape[0] == 100
        assert X_test.shape[0] == 500
        assert X_train.shape[1] == 50
