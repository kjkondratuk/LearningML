"""Tests for Module 03: Pandas Data Wrangling.

Run with: pytest tests/test_exercises.py -v
"""

import numpy as np
import pandas as pd
import pytest

from exercises import feature_summary, group_aggregate, load_and_clean, merge_datasets


# ---------------------------------------------------------------------------
# load_and_clean
# ---------------------------------------------------------------------------

class TestLoadAndClean:
    def _make_messy_df(self):
        return pd.DataFrame({
            " First Name ": ["Alice", "Bob", "Charlie", "Alice", np.nan],
            "Age": [30, np.nan, 25, 30, 40],
            " Score ": [85.0, 92.0, np.nan, 85.0, 78.0],
        })

    def test_column_names_normalized(self):
        df = load_and_clean(self._make_messy_df())
        assert list(df.columns) == ["first_name", "age", "score"]

    def test_duplicates_removed(self):
        raw = self._make_messy_df()
        df = load_and_clean(raw)
        # Row 0 and row 3 are duplicates after cleaning (Alice, 30, 85.0)
        assert len(df) < len(raw)

    def test_numeric_nans_filled_with_median(self):
        df = load_and_clean(self._make_messy_df())
        assert df["age"].isna().sum() == 0
        assert df["score"].isna().sum() == 0

    def test_string_nans_filled_with_unknown(self):
        df = load_and_clean(self._make_messy_df())
        assert "unknown" in df["first_name"].values

    def test_index_is_reset(self):
        df = load_and_clean(self._make_messy_df())
        assert list(df.index) == list(range(len(df)))

    def test_does_not_modify_input(self):
        raw = self._make_messy_df()
        original_cols = list(raw.columns)
        load_and_clean(raw)
        assert list(raw.columns) == original_cols


# ---------------------------------------------------------------------------
# feature_summary
# ---------------------------------------------------------------------------

class TestFeatureSummary:
    def _make_df(self):
        return pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, np.nan],
            "b": [10, 20, 30, 40, 50],
            "name": ["x", "y", "z", "w", "v"],
        })

    def test_only_numeric_columns(self):
        summary = feature_summary(self._make_df())
        assert "name" not in summary.index
        assert "a" in summary.index
        assert "b" in summary.index

    def test_has_required_stats(self):
        summary = feature_summary(self._make_df())
        required = ["count", "mean", "std", "min", "max", "null_count", "null_pct"]
        for stat in required:
            assert stat in summary.columns, f"Missing column: {stat}"

    def test_null_count_correct(self):
        summary = feature_summary(self._make_df())
        assert summary.loc["a", "null_count"] == 1
        assert summary.loc["b", "null_count"] == 0

    def test_null_pct_correct(self):
        summary = feature_summary(self._make_df())
        assert abs(summary.loc["a", "null_pct"] - 20.0) < 0.1

    def test_mean_correct(self):
        summary = feature_summary(self._make_df())
        assert abs(summary.loc["a", "mean"] - 2.5) < 0.01
        assert abs(summary.loc["b", "mean"] - 30.0) < 0.01

    def test_count_excludes_nans(self):
        summary = feature_summary(self._make_df())
        assert summary.loc["a", "count"] == 4
        assert summary.loc["b", "count"] == 5


# ---------------------------------------------------------------------------
# group_aggregate
# ---------------------------------------------------------------------------

class TestGroupAggregate:
    def _make_df(self):
        return pd.DataFrame({
            "category": ["A", "A", "A", "B", "B", "C"],
            "value": [10.0, 20.0, 30.0, 100.0, 200.0, 50.0],
        })

    def test_correct_groups(self):
        result = group_aggregate(self._make_df(), "category", "value")
        assert set(result.index) == {"A", "B", "C"}

    def test_has_required_columns(self):
        result = group_aggregate(self._make_df(), "category", "value")
        for col in ["mean", "median", "std", "count"]:
            assert col in result.columns

    def test_mean_values(self):
        result = group_aggregate(self._make_df(), "category", "value")
        assert abs(result.loc["A", "mean"] - 20.0) < 0.01
        assert abs(result.loc["B", "mean"] - 150.0) < 0.01

    def test_count_values(self):
        result = group_aggregate(self._make_df(), "category", "value")
        assert result.loc["A", "count"] == 3
        assert result.loc["B", "count"] == 2
        assert result.loc["C", "count"] == 1

    def test_sorted_by_mean_descending(self):
        result = group_aggregate(self._make_df(), "category", "value")
        means = result["mean"].values
        assert all(means[i] >= means[i + 1] for i in range(len(means) - 1))

    def test_index_is_group_col(self):
        result = group_aggregate(self._make_df(), "category", "value")
        assert result.index.name == "category"


# ---------------------------------------------------------------------------
# merge_datasets
# ---------------------------------------------------------------------------

class TestMergeDatasets:
    def _make_left(self):
        return pd.DataFrame({
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "score": [85, 92, 78],
        })

    def _make_right(self):
        return pd.DataFrame({
            "id": [2, 3, 4],
            "name": ["Robert", "Charles", "Diana"],
            "grade": ["A", "B", "A"],
        })

    def test_inner_join_length(self):
        result = merge_datasets(self._make_left(), self._make_right(), on="id", how="inner")
        assert len(result) == 2  # ids 2 and 3

    def test_left_join_keeps_all_left(self):
        result = merge_datasets(self._make_left(), self._make_right(), on="id", how="left")
        assert len(result) == 3  # ids 1, 2, 3

    def test_outer_join_keeps_all(self):
        result = merge_datasets(self._make_left(), self._make_right(), on="id", how="outer")
        assert len(result) == 4  # ids 1, 2, 3, 4

    def test_left_values_preferred_for_shared_columns(self):
        result = merge_datasets(self._make_left(), self._make_right(), on="id", how="inner")
        # "name" exists in both — left values should be kept
        assert "Bob" in result["name"].values
        assert "Robert" not in result["name"].values

    def test_sorted_by_join_key(self):
        result = merge_datasets(self._make_left(), self._make_right(), on="id", how="outer")
        assert list(result["id"]) == sorted(result["id"])

    def test_index_is_reset(self):
        result = merge_datasets(self._make_left(), self._make_right(), on="id", how="inner")
        assert list(result.index) == list(range(len(result)))

    def test_no_duplicate_name_columns(self):
        result = merge_datasets(self._make_left(), self._make_right(), on="id", how="inner")
        name_cols = [c for c in result.columns if "name" in c.lower()]
        assert len(name_cols) == 1, f"Expected 1 name column, got: {name_cols}"

    def test_outer_join_fills_left_name_with_right(self):
        result = merge_datasets(self._make_left(), self._make_right(), on="id", how="outer")
        # id=4 only exists in right, so name should come from right
        row_4 = result[result["id"] == 4].iloc[0]
        assert row_4["name"] == "Diana"
