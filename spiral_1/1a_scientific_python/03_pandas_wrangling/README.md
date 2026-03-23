# Module 03: Pandas Data Wrangling

> **Time estimate:** 4--6 hours
> **Prerequisites:** Module 02 complete
> **Math prep:** [StatQuest: Histograms](https://www.youtube.com/watch?v=qBigTkBLU6g) and [StatQuest: Mean, Variance, Std Dev](https://www.youtube.com/watch?v=SzZ6GpcfoQY)

## Why Pandas

Real-world data arrives messy: CSV files with missing values, inconsistent column names, duplicate rows, mixed types, and dates in twelve different formats. Before any ML model touches the data, you need to clean it, explore it, and reshape it. That's Pandas.

Pandas is to tabular data what NumPy is to numeric arrays. Under the hood, it *is* NumPy---a DataFrame is a collection of named NumPy arrays (Series) with aligned indexes and a massive API for slicing, grouping, merging, and reshaping.

## Concepts

### 1. DataFrames and Series

A DataFrame is a 2D labeled table. Each column is a Series (a 1D labeled array). The index labels the rows.

> **Go parallel:** Think of a DataFrame as a `map[string][]any` where all the slices have the same length and the map keys are column names. But unlike a Go map, column order is preserved and there's a rich API for aligned operations. Alternatively, think of it as a database table you can query in Python.

**Key operations:**
- `df["column"]` or `df.column` — select a column (returns a Series)
- `df[["col1", "col2"]]` — select multiple columns (returns a DataFrame)
- `df.loc[row_label, col_label]` — label-based indexing
- `df.iloc[row_idx, col_idx]` — integer position-based indexing
- `df.head()`, `df.info()`, `df.describe()` — quick inspection

### 2. Handling Missing Data

Missing values (`NaN`, `None`, `NaT`) are everywhere in real data. Pandas has a consistent API for detecting and handling them.

**Strategies:**
- **Drop rows:** `df.dropna()` — simple but loses data
- **Fill with a value:** `df.fillna(0)` or `df.fillna(df.mean())` — keeps rows but introduces assumptions
- **Interpolate:** `df.interpolate()` — good for time series
- **Forward/backward fill:** `df.ffill()`, `df.bfill()` — propagates last known value

The right strategy depends on context. Dropping is fine if NaNs are rare. Filling with the mean is common for numeric features. You'll develop judgment through the exercises.

### 3. GroupBy: Split-Apply-Combine

GroupBy is Pandas' most powerful pattern. It splits data by some key, applies a function to each group, and combines the results.

> **Go parallel:** This is like `map[GroupKey][]Row` where you build the map with a loop, then iterate over each group to compute something. Pandas does it in one line: `df.groupby("category")["value"].mean()`.

```python
# Average sale price per neighborhood
df.groupby("neighborhood")["sale_price"].mean()

# Multiple aggregations at once
df.groupby("neighborhood")["sale_price"].agg(["mean", "median", "count"])
```

### 4. Merging and Joining

Combining data from multiple tables is fundamental. Pandas merge is SQL's JOIN.

```python
# Inner join on a shared column
merged = pd.merge(orders, customers, on="customer_id", how="inner")

# Left join — keep all rows from the left table
merged = pd.merge(orders, customers, on="customer_id", how="left")
```

> **Go parallel:** There's no built-in join in Go. You'd write nested loops or build a map for lookup. Pandas merge handles all the index alignment, missing value insertion, and duplicate handling for you.

**Join types:**
- `inner` — only rows with matching keys in both tables
- `left` — all rows from left, matched rows from right (NaN where no match)
- `right` — all rows from right, matched rows from left
- `outer` — all rows from both tables

### 5. Method Chaining

Pandas encourages a functional style where you chain operations. This reads top-to-bottom like a data pipeline.

```python
result = (
    df
    .dropna(subset=["price"])
    .query("price > 0")
    .assign(log_price=lambda d: np.log(d["price"]))
    .groupby("category")["log_price"]
    .mean()
    .sort_values(ascending=False)
)
```

Each method returns a new DataFrame, so you can chain without intermediate variables. This style is idiomatic Pandas and you'll see it throughout ML codebases.

## Exercises

Implement the stubs in `exercises.py`. Run `pytest tests/test_exercises.py -v`.

| Function | What It Practices |
|----------|------------------|
| `load_and_clean` | Reading data, null handling, type conversion, deduplication |
| `feature_summary` | Descriptive stats, value counts, conditional logic |
| `group_aggregate` | GroupBy, multiple aggregations, sorting |
| `merge_datasets` | Merging with different join types, handling key mismatches |

## Mini-Project: Clean Messy Data

In `mini_project.py`, implement `clean_messy_data` — a full cleaning pipeline that takes deliberately terrible data and produces a clean, analysis-ready DataFrame. Run `pytest tests/test_mini_project.py -v`.

The messy data has:
- Inconsistent column names (mixed case, extra whitespace)
- Missing values in various columns
- Duplicate rows
- String columns that should be numeric (e.g., "$1,234.56")
- Date strings in multiple formats
- Outlier values that are clearly data entry errors

## Do It the Wrong Way Challenge

Pick one of your exercise functions (e.g., `group_aggregate`) and implement it two ways:

1. **Pure Python:** Use only dicts, lists, and loops. No Pandas.
2. **Pandas:** Use DataFrame operations.

Compare:
- Lines of code
- Readability
- Performance on 100,000 rows (use `time.perf_counter()`)

```python
import time
import pandas as pd
import numpy as np

# Generate test data
n = 100_000
df = pd.DataFrame({
    "category": np.random.choice(["A", "B", "C", "D"], n),
    "value": np.random.randn(n) * 100,
})

# Your dict/loop version
start = time.perf_counter()
result_slow = group_aggregate_with_dicts(df.to_dict("records"))
slow_time = time.perf_counter() - start

# Your Pandas version
start = time.perf_counter()
result_fast = group_aggregate(df, "category", "value")
fast_time = time.perf_counter() - start

print(f"Dicts/loops: {slow_time:.4f}s | Pandas: {fast_time:.4f}s | Speedup: {slow_time/fast_time:.0f}x")
```

## Style Notes

- **Use `.copy()` when you slice** to avoid the `SettingWithCopyWarning`. `subset = df[df["col"] > 5].copy()`.
- **Method chaining** is preferred over sequential variable reassignment. It makes the pipeline explicit and each step inspectable.
- **Name your aggregations.** `df.groupby("x")["y"].agg(mean_y="mean", count_y="count")` is clearer than unnamed results.
- **Use `.query()` for complex filters.** `df.query("price > 100 and category == 'A'")` reads better than boolean masking for multi-condition filters.
- **Check `.dtypes` early.** Many Pandas bugs come from columns being `object` (string) when they should be numeric. Use `df.dtypes` and `pd.to_numeric()` proactively.
