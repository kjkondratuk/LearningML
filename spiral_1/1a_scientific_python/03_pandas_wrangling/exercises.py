"""Module 03: Pandas Data Wrangling — Exercise Stubs

Implement each function below until all tests in tests/test_exercises.py pass.
"""

import pandas as pd


def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a raw DataFrame by applying standard preprocessing steps.

    Steps to implement:
    1. Strip whitespace from column names and convert them to lowercase with
       underscores replacing spaces.
    2. Drop completely duplicate rows.
    3. For numeric columns, fill NaN values with the column median.
    4. For string/object columns, fill NaN values with the string "unknown".
    5. Reset the index (drop the old index).

    Args:
        df: Raw input DataFrame (do not modify in place).

    Returns:
        A new, cleaned DataFrame.
    """
    raise NotImplementedError


def feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a summary of each numeric column in the DataFrame.

    For each numeric column, compute:
    - count: number of non-null values
    - mean: arithmetic mean
    - std: standard deviation
    - min: minimum value
    - max: maximum value
    - null_count: number of null values
    - null_pct: percentage of null values (0 to 100)

    Args:
        df: Input DataFrame with mixed column types.

    Returns:
        A DataFrame where each row is a numeric column from the input,
        and the columns are the summary statistics listed above.
        The index should be the column names.
    """
    raise NotImplementedError


def group_aggregate(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Group by a column and compute aggregations on a value column.

    Compute per group:
    - mean: mean of value_col
    - median: median of value_col
    - std: standard deviation of value_col
    - count: number of rows

    Sort the result by mean in descending order.

    Args:
        df: Input DataFrame.
        group_col: Column name to group by.
        value_col: Column name to aggregate.

    Returns:
        A DataFrame with group_col as the index and columns [mean, median, std, count],
        sorted by mean descending.
    """
    raise NotImplementedError


def merge_datasets(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    how: str = "inner",
) -> pd.DataFrame:
    """Merge two DataFrames and clean up the result.

    Steps:
    1. Merge left and right on the specified column using the specified join type.
    2. For any columns that appear in both DataFrames (besides the join key),
       prefer the left value: fill NaN in the left version with the right version,
       then drop the right version.
    3. Sort by the join key.
    4. Reset the index.

    Args:
        left: Left DataFrame.
        right: Right DataFrame.
        on: Column name to join on.
        how: Join type — "inner", "left", "right", or "outer".

    Returns:
        A merged, cleaned DataFrame.
    """
    raise NotImplementedError
