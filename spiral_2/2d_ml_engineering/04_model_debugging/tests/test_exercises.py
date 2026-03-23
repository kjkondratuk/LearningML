"""
Tests for Model Debugging exercises.
"""

import numpy as np
import pytest

from exercises import (
    training_diagnostics,
    detect_data_leakage,
)


class TestTrainingDiagnostics:
    def test_overfitting(self):
        train_losses = [1.0, 0.5, 0.1, 0.01, 0.001]
        val_losses = [1.0, 0.8, 0.9, 1.0, 1.2]
        diag = training_diagnostics(train_losses, val_losses)
        assert "overfit" in diag.lower()

    def test_underfitting(self):
        train_losses = [2.0, 1.9, 1.85, 1.82, 1.80]
        val_losses = [2.1, 2.0, 1.95, 1.92, 1.90]
        diag = training_diagnostics(train_losses, val_losses)
        assert "underfit" in diag.lower()

    def test_good_fit(self):
        train_losses = [1.0, 0.5, 0.2, 0.1, 0.08]
        val_losses = [1.1, 0.55, 0.25, 0.15, 0.12]
        diag = training_diagnostics(train_losses, val_losses)
        assert "good" in diag.lower() or "fit" in diag.lower()


class TestDataLeakage:
    def test_detects_duplicates(self):
        X_train = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        X_test = np.array([[3, 4], [7, 8]], dtype=float)
        report = detect_data_leakage(X_train, X_test)
        assert report['duplicate_rows'] > 0

    def test_no_leakage(self):
        X_train = np.array([[1, 2], [3, 4]], dtype=float)
        X_test = np.array([[5, 6], [7, 8]], dtype=float)
        report = detect_data_leakage(X_train, X_test)
        assert report['duplicate_rows'] == 0
