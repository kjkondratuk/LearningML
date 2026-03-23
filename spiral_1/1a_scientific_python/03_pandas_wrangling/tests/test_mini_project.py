"""Tests for Module 03: Mini-Project — Clean Messy Data.

Run with: pytest tests/test_mini_project.py -v
"""

import numpy as np
import pandas as pd
import pytest

from mini_project import clean_messy_data


def _make_messy_df():
    """Create a deliberately messy DataFrame for testing."""
    return pd.DataFrame({
        " Price ": ["$1,234.56", "$999.99", 500.0, np.nan, "$1,234.56", "$0.99", "$50"],
        "CATEGORY": ["Electronics", "ELECTRONICS", "clothing", "Home", "Electronics", "clothing", np.nan],
        "  date_Sold": [
            "2023-01-15",
            "01/15/2023",
            "January 15, 2023",
            "2023-06-01",
            "2023-01-15",  # duplicate row (same as first after cleaning)
            "03/20/2023",
            "2023-12-25",
        ],
        " Quantity ": [5, 10, -3, 2000, 5, 0, 8],
    })


class TestCleanMessyData:
    def test_column_names_normalized(self):
        df = clean_messy_data(_make_messy_df())
        for col in df.columns:
            assert col == col.lower().strip()
            assert "  " not in col

    def test_price_is_float(self):
        df = clean_messy_data(_make_messy_df())
        assert df["price"].dtype in [np.float64, np.float32]

    def test_price_values_parsed(self):
        df = clean_messy_data(_make_messy_df())
        prices = sorted(df["price"].dropna().unique())
        assert 0.99 in prices or any(abs(p - 0.99) < 0.01 for p in prices)
        assert any(abs(p - 1234.56) < 0.01 for p in prices)

    def test_dates_are_datetime(self):
        df = clean_messy_data(_make_messy_df())
        assert pd.api.types.is_datetime64_any_dtype(df["date_sold"])

    def test_no_duplicate_rows(self):
        raw = _make_messy_df()
        df = clean_messy_data(raw)
        assert len(df) < len(raw), "Duplicates should have been removed"
        assert df.duplicated().sum() == 0

    def test_category_lowercase(self):
        df = clean_messy_data(_make_messy_df())
        for val in df["category"].dropna():
            assert val == val.lower(), f"Category '{val}' is not lowercase"

    def test_quantity_no_negatives(self):
        df = clean_messy_data(_make_messy_df())
        assert (df["quantity"] >= 0).all()

    def test_quantity_outliers_handled(self):
        df = clean_messy_data(_make_messy_df())
        # Values > 1000 should have been clipped to NaN then filled
        assert (df["quantity"] <= 1000).all()

    def test_no_numeric_nans(self):
        df = clean_messy_data(_make_messy_df())
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert df[col].isna().sum() == 0, f"Column '{col}' still has NaN values"

    def test_index_is_reset(self):
        df = clean_messy_data(_make_messy_df())
        assert list(df.index) == list(range(len(df)))

    def test_does_not_modify_input(self):
        raw = _make_messy_df()
        original_shape = raw.shape
        clean_messy_data(raw)
        assert raw.shape == original_shape
