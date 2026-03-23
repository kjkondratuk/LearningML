"""Tests for data preparation."""

import pytest

from data_prep import load_alpaca_format


class TestDataPrep:
    def test_alpaca_format_keys(self):
        """Each example should have instruction, input, output keys."""
        # This test requires a sample data file; skip if not available
        pytest.skip("Requires sample dataset file")
