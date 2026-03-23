"""
Tests for Module 02: Mini-Project — Predict Housing Prices

Run: pytest tests/test_mini_project.py -v
"""

import numpy as np
import pytest

from mini_project import predict_housing_prices


class TestPredictHousingPrices:
    def setup_method(self):
        """Create a synthetic housing dataset with known linear relationship."""
        rng = np.random.RandomState(42)
        n_samples = 200
        # 3 features: size, bedrooms, age
        self.X = rng.randn(n_samples, 3)
        # True relationship: price = 50*size + 20*bedrooms - 10*age + 100 + noise
        self.y = (
            50 * self.X[:, 0]
            + 20 * self.X[:, 1]
            - 10 * self.X[:, 2]
            + 100
            + rng.randn(n_samples) * 5
        )

    def test_returns_dict(self):
        result = predict_housing_prices(self.X, self.y)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = predict_housing_prices(self.X, self.y)
        for key in ["gd_mse", "sklearn_mse", "gd_weights", "sklearn_coef", "loss_history"]:
            assert key in result, f"Missing key: {key}"

    def test_mse_is_reasonable(self):
        result = predict_housing_prices(self.X, self.y)
        # On a nearly-linear dataset, both models should achieve low MSE
        assert result["sklearn_mse"] < 100.0, "sklearn MSE is too high for this data"

    def test_loss_history_decreases(self):
        result = predict_housing_prices(self.X, self.y)
        losses = result["loss_history"]
        assert len(losses) > 1
        assert losses[-1] < losses[0], "Training loss should decrease"

    def test_weights_shapes(self):
        result = predict_housing_prices(self.X, self.y)
        assert result["gd_weights"].shape == (3,)
        assert result["sklearn_coef"].shape == (3,)

    def test_different_random_state(self):
        r1 = predict_housing_prices(self.X, self.y, random_state=1)
        r2 = predict_housing_prices(self.X, self.y, random_state=2)
        # Different splits should give different (but both reasonable) results
        assert r1["sklearn_mse"] != r2["sklearn_mse"]
