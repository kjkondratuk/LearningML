"""
Tests for the Production Pipeline capstone.
"""

import pytest
import os

from pipeline import (
    load_config,
    set_reproducibility,
)


class TestConfig:
    def test_load_config(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.yaml"
        )
        if os.path.exists(config_path):
            config = load_config(config_path)
            assert "experiment" in config
            assert "data" in config
            assert "training" in config
            assert "reproducibility" in config


class TestReproducibility:
    def test_deterministic_output(self):
        """Same seed should produce same random numbers."""
        import numpy as np
        set_reproducibility(42)
        a = np.random.randn(10)
        set_reproducibility(42)
        b = np.random.randn(10)
        np.testing.assert_array_equal(a, b)


class TestPipelineIntegration:
    def test_config_has_all_required_fields(self):
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "config.yaml"
        )
        if os.path.exists(config_path):
            config = load_config(config_path)
            assert "seed" in config.get("reproducibility", {})
            assert "n_epochs" in config.get("training", {})
            assert "batch_size" in config.get("data", {})
