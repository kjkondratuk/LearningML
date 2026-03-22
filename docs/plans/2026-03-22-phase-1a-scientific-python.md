# Phase 1A: Python for Scientific Computing — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build fluency with Python's scientific computing stack, bridging from Go developer mental models to idiomatic Python.

**Architecture:** Five modules, each producing working code in `spiral_1/1a_scientific_python/`. Jupyter notebooks for exploration, `.py` files for reusable code. Each module builds on the previous.

**Tech Stack:** Python 3.14, NumPy, Pandas, Matplotlib, Seaborn, Jupyter, pytest

---

### Task 1: Project Setup & Dependencies

**Files:**
- Modify: `pyproject.toml`
- Create: `spiral_1/__init__.py`
- Create: `spiral_1/1a_scientific_python/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/spiral_1/__init__.py`
- Create: `tests/spiral_1/test_1a/__init__.py`

**Step 1: Add dependencies to pyproject.toml**

```toml
[project]
name = "learningml"
version = "0.1.0"
description = "ML learning curriculum — from basics to research frontier"
requires-python = ">=3.14"
dependencies = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "jupyter",
    "ipykernel",
    "scikit-learn",
]

[project.optional-dependencies]
dev = [
    "pytest",
    "ruff",
]
```

**Step 2: Install dependencies**

Run: `cd /Users/kkondratuk/PycharmProjects/LearningML && uv sync && uv pip install -e ".[dev]"`

**Step 3: Create directory structure**

```bash
mkdir -p spiral_1/1a_scientific_python
mkdir -p tests/spiral_1/test_1a
touch spiral_1/__init__.py
touch spiral_1/1a_scientific_python/__init__.py
touch tests/__init__.py
touch tests/spiral_1/__init__.py
touch tests/spiral_1/test_1a/__init__.py
```

**Step 4: Verify setup**

Run: `uv run python -c "import numpy; import pandas; import matplotlib; print('All imports OK')"`
Expected: `All imports OK`

**Step 5: Commit**

```bash
git add pyproject.toml spiral_1/ tests/
git commit -m "setup: add scientific python dependencies and directory structure"
```

---

### Task 2: Python Idioms for Go Developers

**Files:**
- Create: `spiral_1/1a_scientific_python/01_python_idioms.py`
- Create: `tests/spiral_1/test_1a/test_01_python_idioms.py`

This module covers the Python patterns that differ most from Go. Each exercise has a Go-equivalent comment explaining what this replaces.

**Step 1: Write the test file**

```python
"""Tests for Python idioms exercises.

Each test verifies understanding of a Python pattern that
has no direct Go equivalent.
"""
import pytest
from spiral_1.1a_scientific_python.exercises_01 import (
    fibonacci_generator,
    retry,
    ManagedResource,
    transform_pipeline,
    DataPoint,
)


class TestGenerators:
    """Go equivalent: channels + goroutines for lazy sequences."""

    def test_fibonacci_first_10(self):
        gen = fibonacci_generator()
        first_10 = [next(gen) for _ in range(10)]
        assert first_10 == [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    def test_fibonacci_is_lazy(self):
        """Generator should not compute ahead — verify it yields one at a time."""
        gen = fibonacci_generator()
        assert next(gen) == 0
        assert next(gen) == 1
        assert next(gen) == 1


class TestDecorators:
    """Go equivalent: middleware / wrapper functions, but clunkier without generics."""

    def test_retry_succeeds_eventually(self):
        call_count = 0

        @retry(max_attempts=3)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("not yet")
            return "success"

        assert flaky_function() == "success"
        assert call_count == 3

    def test_retry_exhausts_attempts(self):
        @retry(max_attempts=2)
        def always_fails():
            raise ValueError("nope")

        with pytest.raises(ValueError, match="nope"):
            always_fails()


class TestContextManagers:
    """Go equivalent: defer, but context managers handle setup AND teardown."""

    def test_managed_resource_lifecycle(self):
        events = []
        with ManagedResource("test", events) as resource:
            events.append(f"using:{resource.name}")
        assert events == ["open:test", "using:test", "close:test"]

    def test_managed_resource_closes_on_error(self):
        events = []
        with pytest.raises(RuntimeError):
            with ManagedResource("test", events):
                raise RuntimeError("boom")
        assert "close:test" in events


class TestComprehensionsAndPipelines:
    """Go equivalent: for loops with append. Python is more expressive here."""

    def test_transform_pipeline(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = transform_pipeline(data)
        # Filter evens, square them, keep those > 10
        assert result == [16, 36, 64, 100]


class TestDataclasses:
    """Go equivalent: structs. Dataclasses add __init__, __repr__, __eq__ for free."""

    def test_datapoint_creation(self):
        dp = DataPoint(x=1.0, y=2.0, label="a")
        assert dp.x == 1.0
        assert dp.label == "a"

    def test_datapoint_equality(self):
        dp1 = DataPoint(x=1.0, y=2.0, label="a")
        dp2 = DataPoint(x=1.0, y=2.0, label="a")
        assert dp1 == dp2
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/spiral_1/test_1a/test_01_python_idioms.py -v`
Expected: FAIL — `ModuleNotFoundError` (exercises_01 doesn't exist yet)

**Step 3: Implement the exercises**

```python
"""Python idioms for Go developers.

Each section implements a pattern with comments explaining
the Go equivalent and why Python does it differently.
"""
from dataclasses import dataclass
from typing import Generator
import functools


# --- Generators ---
# Go equivalent: You'd use a channel + goroutine to produce values lazily.
# Python generators do the same thing with simpler syntax.
# `yield` suspends the function and resumes on next().

def fibonacci_generator() -> Generator[int, None, None]:
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


# --- Decorators ---
# Go equivalent: Wrapping a function/handler with middleware.
# In Go you'd write: handler = retryMiddleware(handler)
# Python decorators are syntactic sugar for the same pattern.

def retry(max_attempts: int = 3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for _ in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
            raise last_exception
        return wrapper
    return decorator


# --- Context Managers ---
# Go equivalent: defer. But context managers handle both setup AND teardown,
# and they compose with `with` blocks for clear scoping.

class ManagedResource:
    def __init__(self, name: str, events: list[str]):
        self.name = name
        self.events = events

    def __enter__(self):
        self.events.append(f"open:{self.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.events.append(f"close:{self.name}")
        return False  # Don't suppress exceptions


# --- Comprehensions & Pipelines ---
# Go equivalent: for loops with append and if-statements.
# Python comprehensions express the same logic more concisely.
# Go devs often find this hard to read at first — it becomes natural.

def transform_pipeline(data: list[int]) -> list[int]:
    return [x * x for x in data if x % 2 == 0 and x * x > 10]


# --- Dataclasses ---
# Go equivalent: struct with manually written constructors.
# Dataclasses auto-generate __init__, __repr__, __eq__, and more.

@dataclass
class DataPoint:
    x: float
    y: float
    label: str
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/spiral_1/test_1a/test_01_python_idioms.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add spiral_1/1a_scientific_python/exercises_01.py tests/spiral_1/test_1a/test_01_python_idioms.py
git commit -m "feat(1a): python idioms for go developers — generators, decorators, context managers"
```

---

### Task 3: NumPy Fundamentals

**Files:**
- Create: `spiral_1/1a_scientific_python/02_numpy_fundamentals.ipynb`
- Create: `spiral_1/1a_scientific_python/exercises_02.py`
- Create: `tests/spiral_1/test_1a/test_02_numpy.py`

**Step 1: Write the test file**

```python
"""Tests for NumPy exercises.

Focus on the core operations you'll use constantly in ML:
array creation, indexing, broadcasting, vectorization.
"""
import numpy as np
import pytest
from spiral_1.1a_scientific_python.exercises_02 import (
    normalize_array,
    euclidean_distance,
    moving_average,
    one_hot_encode,
    batch_dot_product,
)


class TestNormalization:
    """Normalization (scaling to mean=0, std=1) is used in nearly every ML pipeline."""

    def test_normalize_basic(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = normalize_array(arr)
        assert np.isclose(result.mean(), 0.0, atol=1e-10)
        assert np.isclose(result.std(), 1.0, atol=1e-10)

    def test_normalize_2d_columns(self):
        """Normalize each column independently — common for feature matrices."""
        arr = np.array([[1.0, 100.0], [2.0, 200.0], [3.0, 300.0]])
        result = normalize_array(arr, axis=0)
        assert np.allclose(result.mean(axis=0), [0.0, 0.0], atol=1e-10)
        assert np.allclose(result.std(axis=0), [1.0, 1.0], atol=1e-10)


class TestEuclideanDistance:
    """Distance metrics are the backbone of clustering and nearest-neighbor methods."""

    def test_distance_same_point(self):
        a = np.array([1.0, 2.0, 3.0])
        assert euclidean_distance(a, a) == 0.0

    def test_distance_known_value(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        assert euclidean_distance(a, b) == 5.0

    def test_distance_batch(self):
        """Compute distance from one point to many points — no Python loops."""
        point = np.array([0.0, 0.0])
        points = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 1.0]])
        result = euclidean_distance(point, points)
        expected = np.array([5.0, 1.0, 1.0])
        assert np.allclose(result, expected)


class TestMovingAverage:
    """Smoothing signals — you'll use this in loss curve visualization."""

    def test_moving_average_window_3(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(data, window=3)
        expected = np.array([2.0, 3.0, 4.0])
        assert np.allclose(result, expected)


class TestOneHotEncode:
    """Converting categorical labels to vectors — fundamental for classification."""

    def test_one_hot_basic(self):
        labels = np.array([0, 2, 1])
        result = one_hot_encode(labels, num_classes=3)
        expected = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ])
        assert np.array_equal(result, expected)


class TestBatchDotProduct:
    """Batch operations are critical for ML — process many samples at once."""

    def test_batch_dot(self):
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        result = batch_dot_product(a, b)
        expected = np.array([17, 53])  # [1*5+2*6, 3*7+4*8]
        assert np.array_equal(result, expected)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/spiral_1/test_1a/test_02_numpy.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement the exercises**

```python
"""NumPy fundamentals for ML.

Key mental model shift from Go: stop thinking in loops, start thinking
in array operations. NumPy runs C under the hood — vectorized code
is 10-100x faster than Python for-loops.

Go equivalent of NumPy: there isn't one. Go's math libraries work
element-by-element. NumPy operates on entire arrays at once.
"""
import numpy as np


def normalize_array(arr: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Normalize to mean=0, std=1 (z-score normalization).

    This is the most common preprocessing step in ML.
    Formula: (x - mean) / std
    """
    mean = arr.mean(axis=axis, keepdims=True)
    std = arr.std(axis=axis, keepdims=True)
    return (arr - mean) / std


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray | float:
    """Compute Euclidean distance between point(s).

    Works for:
    - Two 1D arrays (returns scalar)
    - One 1D array and one 2D array (returns array of distances)

    Key insight: broadcasting lets us compute many distances at once
    without writing a loop.
    """
    diff = a - b
    return np.sqrt(np.sum(diff ** 2, axis=-1))


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Compute moving average using np.convolve.

    You'll use this to smooth noisy loss curves during training.
    """
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer labels to one-hot vectors.

    Example: label 2 with 4 classes -> [0, 0, 1, 0]

    This is how we represent categorical targets for neural networks.
    No loops — use fancy indexing.
    """
    result = np.zeros((len(labels), num_classes), dtype=int)
    result[np.arange(len(labels)), labels] = 1
    return result


def batch_dot_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute dot products for each pair of row vectors.

    This is a building block for matrix multiplication,
    which is THE core operation in neural networks.
    """
    return np.sum(a * b, axis=1)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/spiral_1/test_1a/test_02_numpy.py -v`
Expected: All 7 tests PASS

**Step 5: Create the companion notebook**

Create `spiral_1/1a_scientific_python/02_numpy_fundamentals.ipynb` with cells covering:
1. Array creation: `np.array`, `np.zeros`, `np.ones`, `np.arange`, `np.linspace`
2. Indexing and slicing: basic, fancy, boolean indexing
3. Broadcasting rules: visualize how shapes align
4. Vectorization speed comparison: loop vs NumPy for 1M elements
5. Reshape, transpose, and axis operations

**Step 6: Commit**

```bash
git add spiral_1/1a_scientific_python/exercises_02.py spiral_1/1a_scientific_python/02_numpy_fundamentals.ipynb tests/spiral_1/test_1a/test_02_numpy.py
git commit -m "feat(1a): numpy fundamentals — normalization, distance, broadcasting, one-hot"
```

---

### Task 4: Pandas for Data Wrangling

**Files:**
- Create: `spiral_1/1a_scientific_python/03_pandas_wrangling.ipynb`
- Create: `spiral_1/1a_scientific_python/exercises_03.py`
- Create: `tests/spiral_1/test_1a/test_03_pandas.py`

**Step 1: Write the test file**

```python
"""Tests for Pandas exercises.

Pandas is to tabular data what NumPy is to numerical arrays.
You'll use it for every data loading/cleaning task.
"""
import pandas as pd
import numpy as np
import pytest
from spiral_1.1a_scientific_python.exercises_03 import (
    load_and_clean,
    feature_summary,
    group_aggregate,
    merge_datasets,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", None, "Eve"],
        "age": [25, 30, None, 22, 28],
        "score": [85.0, 92.0, 78.0, 95.0, None],
        "group": ["A", "B", "A", "B", "A"],
    })


class TestLoadAndClean:
    """Real datasets are always messy. Cleaning is 80% of the work."""

    def test_drops_rows_with_null_name(self, sample_df):
        result = load_and_clean(sample_df)
        assert result["name"].isna().sum() == 0
        assert len(result) == 4

    def test_fills_numeric_nulls_with_median(self, sample_df):
        result = load_and_clean(sample_df)
        assert result["age"].isna().sum() == 0
        assert result["score"].isna().sum() == 0


class TestFeatureSummary:
    """Quick statistics of your features — first thing you do with any dataset."""

    def test_summary_has_expected_stats(self, sample_df):
        clean = load_and_clean(sample_df)
        summary = feature_summary(clean, numeric_cols=["age", "score"])
        assert "mean" in summary.index
        assert "std" in summary.index
        assert "missing_pct" in summary.index
        assert list(summary.columns) == ["age", "score"]


class TestGroupAggregate:
    """Groupby is SQL's GROUP BY — essential for understanding data distributions."""

    def test_group_mean(self, sample_df):
        clean = load_and_clean(sample_df)
        result = group_aggregate(clean, group_col="group", agg_col="score", agg_func="mean")
        assert isinstance(result, pd.DataFrame)
        assert "group" in result.columns
        assert "score" in result.columns


class TestMergeDatasets:
    """Combining data from multiple sources — like SQL JOINs."""

    def test_inner_merge(self):
        left = pd.DataFrame({"id": [1, 2, 3], "value_a": [10, 20, 30]})
        right = pd.DataFrame({"id": [2, 3, 4], "value_b": [200, 300, 400]})
        result = merge_datasets(left, right, on="id", how="inner")
        assert len(result) == 2
        assert list(result["id"]) == [2, 3]
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/spiral_1/test_1a/test_03_pandas.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement the exercises**

```python
"""Pandas data wrangling for ML pipelines.

Go equivalent: You'd probably use encoding/csv and a bunch of
maps/slices. Pandas gives you a SQL-like API on in-memory tables.
"""
import pandas as pd
import numpy as np


def load_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop rows with null identifiers, fill numeric nulls.

    In ML pipelines, you almost always need to handle missing data.
    Common strategies: drop, fill with mean/median, or impute.
    """
    df = df.copy()
    # Drop rows where name (identifier) is missing
    df = df.dropna(subset=["name"])
    # Fill numeric columns with median (robust to outliers, unlike mean)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    return df.reset_index(drop=True)


def feature_summary(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """Compute summary statistics for numeric features.

    This is the first thing you do when exploring a dataset.
    """
    stats = df[numeric_cols].describe().loc[["mean", "std", "min", "max"]]
    missing = df[numeric_cols].isna().mean().rename("missing_pct")
    return pd.concat([stats, missing.to_frame().T])


def group_aggregate(
    df: pd.DataFrame,
    group_col: str,
    agg_col: str,
    agg_func: str = "mean",
) -> pd.DataFrame:
    """Group by a column and aggregate — like SQL GROUP BY.

    Used constantly to understand data distributions across categories.
    """
    result = df.groupby(group_col)[agg_col].agg(agg_func).reset_index()
    return result


def merge_datasets(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    how: str = "inner",
) -> pd.DataFrame:
    """Merge two DataFrames — like SQL JOIN.

    In ML, you often combine features from different sources.
    """
    return pd.merge(left, right, on=on, how=how)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/spiral_1/test_1a/test_03_pandas.py -v`
Expected: All 5 tests PASS

**Step 5: Create the companion notebook**

Create `spiral_1/1a_scientific_python/03_pandas_wrangling.ipynb` with:
1. Loading CSV/JSON data
2. DataFrame anatomy: index, columns, dtypes
3. Selection: `.loc`, `.iloc`, boolean indexing
4. Method chaining pattern
5. Pivoting and reshaping

**Step 6: Commit**

```bash
git add spiral_1/1a_scientific_python/exercises_03.py spiral_1/1a_scientific_python/03_pandas_wrangling.ipynb tests/spiral_1/test_1a/test_03_pandas.py
git commit -m "feat(1a): pandas data wrangling — cleaning, summary stats, groupby, merging"
```

---

### Task 5: Visualization with Matplotlib & Seaborn

**Files:**
- Create: `spiral_1/1a_scientific_python/04_visualization.ipynb`
- Create: `spiral_1/1a_scientific_python/exercises_04.py`
- Create: `tests/spiral_1/test_1a/test_04_visualization.py`

**Step 1: Write the test file**

```python
"""Tests for visualization exercises.

We can't test visual correctness, but we can verify that
functions produce the right matplotlib objects with expected properties.
"""
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import pytest
from spiral_1.1a_scientific_python.exercises_04 import (
    plot_distribution,
    plot_scatter_with_regression,
    plot_correlation_heatmap,
    plot_training_curves,
)


@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close("all")


class TestDistributionPlot:
    def test_returns_figure_and_axes(self):
        data = np.random.normal(0, 1, 1000)
        fig, ax = plot_distribution(data, title="Test")
        assert isinstance(fig, plt.Figure)
        assert ax.get_title() == "Test"


class TestScatterPlot:
    def test_returns_figure_and_axes(self):
        x = np.linspace(0, 10, 50)
        y = 2 * x + np.random.normal(0, 1, 50)
        fig, ax = plot_scatter_with_regression(x, y)
        assert isinstance(fig, plt.Figure)
        # Should have scatter points and regression line (2 collections/lines)
        assert len(ax.collections) + len(ax.lines) >= 2


class TestHeatmap:
    def test_returns_figure(self):
        data = np.random.randn(5, 5)
        labels = ["a", "b", "c", "d", "e"]
        fig, ax = plot_correlation_heatmap(data, labels)
        assert isinstance(fig, plt.Figure)


class TestTrainingCurves:
    def test_dual_axis_plot(self):
        train_loss = [1.0, 0.8, 0.6, 0.4]
        val_loss = [1.1, 0.9, 0.7, 0.55]
        fig, ax = plot_training_curves(train_loss, val_loss)
        assert isinstance(fig, plt.Figure)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/spiral_1/test_1a/test_04_visualization.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement the exercises**

```python
"""Visualization for ML — using plots as thinking tools.

In ML, you plot constantly:
- Data distributions before modeling
- Feature relationships (scatter, correlation)
- Training/validation curves during training
- Model predictions vs actuals

Go equivalent: there isn't a standard one. Most Go devs export to
a web dashboard. In Python, matplotlib is the universal plotting layer.
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

# Use a clean style
sns.set_theme(style="whitegrid")


def plot_distribution(
    data: np.ndarray,
    title: str = "Distribution",
    bins: int = 30,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a histogram with KDE overlay.

    Use this to check if your features are normally distributed,
    skewed, or have outliers — all of which affect model choice.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data, bins=bins, kde=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    return fig, ax


def plot_scatter_with_regression(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "Scatter with Regression",
) -> tuple[plt.Figure, plt.Axes]:
    """Scatter plot with a linear regression line.

    Use this to visualize relationships between two features,
    or between a feature and target variable.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, alpha=0.6, edgecolors="w", linewidth=0.5)
    # Fit and plot regression line
    coeffs = np.polyfit(x, y, deg=1)
    line_x = np.linspace(x.min(), x.max(), 100)
    line_y = np.polyval(coeffs, line_x)
    ax.plot(line_x, line_y, color="red", linewidth=2, label=f"y = {coeffs[0]:.2f}x + {coeffs[1]:.2f}")
    ax.set_title(title)
    ax.legend()
    return fig, ax


def plot_correlation_heatmap(
    data: np.ndarray,
    labels: list[str],
    title: str = "Correlation Matrix",
) -> tuple[plt.Figure, plt.Axes]:
    """Correlation heatmap — shows which features are related.

    Highly correlated features are often redundant.
    This helps you decide what to keep or drop.
    """
    corr = np.corrcoef(data.T) if data.ndim == 2 else data
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        ax=ax,
    )
    ax.set_title(title)
    return fig, ax


def plot_training_curves(
    train_loss: list[float],
    val_loss: list[float],
    title: str = "Training Curves",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot train vs validation loss over epochs.

    THE most important diagnostic plot during model training.
    - Both decreasing: model is learning
    - Train decreasing, val increasing: overfitting
    - Both flat: model isn't learning (bad lr, architecture, etc.)
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(train_loss) + 1)
    ax.plot(epochs, train_loss, "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs, val_loss, "r-o", label="Val Loss", markersize=4)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    return fig, ax
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/spiral_1/test_1a/test_04_visualization.py -v`
Expected: All 4 tests PASS

**Step 5: Create the companion notebook**

Create `spiral_1/1a_scientific_python/04_visualization.ipynb` with:
1. Matplotlib anatomy: Figure, Axes, subplots
2. Common plot types: line, scatter, bar, histogram
3. Seaborn's statistical plots: boxplot, violinplot, pairplot
4. Customization: colors, labels, legends, styles
5. Saving plots for papers/presentations

**Step 6: Commit**

```bash
git add spiral_1/1a_scientific_python/exercises_04.py spiral_1/1a_scientific_python/04_visualization.ipynb tests/spiral_1/test_1a/test_04_visualization.py
git commit -m "feat(1a): visualization — distributions, scatter, heatmaps, training curves"
```

---

### Task 6: Capstone Project — Exploratory Data Analysis

**Files:**
- Create: `projects/01_eda_capstone/eda_analysis.ipynb`
- Create: `projects/01_eda_capstone/utils.py`

**Step 1: Set up project directory**

```bash
mkdir -p projects/01_eda_capstone
```

**Step 2: Build the EDA notebook**

The capstone ties together everything from Phase 1A. Use a real-world dataset (e.g., the Ames Housing dataset or a Kaggle dataset of your choice).

The notebook should demonstrate:
1. Load data with Pandas
2. Initial inspection: shape, dtypes, `.describe()`, missing values
3. Clean: handle nulls, fix dtypes, remove duplicates
4. Univariate analysis: distributions of key features (matplotlib/seaborn)
5. Bivariate analysis: scatter plots, correlation heatmap
6. Feature engineering: create 2-3 derived features
7. Summary of findings: what would you feed into an ML model?

Extract reusable functions into `utils.py` and import them in the notebook.

**Step 3: Commit**

```bash
git add projects/01_eda_capstone/
git commit -m "feat(1a): capstone EDA project"
```

---

## Completion Criteria

Phase 1A is complete when:
- [ ] All tests pass: `uv run pytest tests/spiral_1/test_1a/ -v`
- [ ] All 4 exercise modules implemented with tests
- [ ] All 3 companion notebooks created and run cleanly
- [ ] Capstone EDA project notebook complete
- [ ] All code committed to git
