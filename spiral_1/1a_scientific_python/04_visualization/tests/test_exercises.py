"""Tests for Module 04: Visualization.

Run with: pytest tests/test_exercises.py -v

These tests verify structural properties of the returned Figure/Axes objects.
They do NOT check visual appearance — that's your job.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from exercises import (
    plot_correlation_heatmap,
    plot_distribution,
    plot_scatter_with_regression,
    plot_training_curves,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# plot_distribution
# ---------------------------------------------------------------------------

class TestPlotDistribution:
    def test_returns_figure_and_axes(self):
        data = np.random.randn(500)
        result = plot_distribution(data)
        assert isinstance(result, tuple) and len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_title_is_set(self):
        data = np.random.randn(200)
        fig, ax = plot_distribution(data, title="My Distribution")
        assert ax.get_title() == "My Distribution"

    def test_xlabel_is_set(self):
        data = np.random.randn(200)
        fig, ax = plot_distribution(data, xlabel="Feature X")
        assert ax.get_xlabel() == "Feature X"

    def test_ylabel_is_set(self):
        data = np.random.randn(200)
        fig, ax = plot_distribution(data, ylabel="Count")
        assert ax.get_ylabel() == "Count"

    def test_has_legend(self):
        data = np.random.randn(200)
        fig, ax = plot_distribution(data)
        legend = ax.get_legend()
        assert legend is not None, "Plot should have a legend"
        labels = [t.get_text() for t in legend.get_texts()]
        assert len(labels) >= 2

    def test_figure_size(self):
        data = np.random.randn(200)
        fig, ax = plot_distribution(data)
        width, height = fig.get_size_inches()
        assert abs(width - 8) < 0.1 and abs(height - 5) < 0.1

    def test_has_patches(self):
        """Histogram should produce rectangular patches."""
        data = np.random.randn(200)
        fig, ax = plot_distribution(data)
        patches = [p for p in ax.patches if isinstance(p, matplotlib.patches.Rectangle)]
        assert len(patches) > 0, "No histogram bars found"


# ---------------------------------------------------------------------------
# plot_scatter_with_regression
# ---------------------------------------------------------------------------

class TestPlotScatterWithRegression:
    def test_returns_figure_and_axes(self):
        x = np.random.randn(100)
        y = 2 * x + np.random.randn(100) * 0.5
        result = plot_scatter_with_regression(x, y)
        assert isinstance(result, tuple) and len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_title_is_set(self):
        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.5
        fig, ax = plot_scatter_with_regression(x, y, title="Price vs Area")
        assert ax.get_title() == "Price vs Area"

    def test_xlabel_and_ylabel(self):
        x = np.random.randn(50)
        y = x * 3
        fig, ax = plot_scatter_with_regression(x, y, xlabel="Area", ylabel="Price")
        assert ax.get_xlabel() == "Area"
        assert ax.get_ylabel() == "Price"

    def test_has_legend(self):
        x = np.random.randn(50)
        y = x + np.random.randn(50) * 0.1
        fig, ax = plot_scatter_with_regression(x, y)
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert len(labels) >= 2

    def test_has_scatter_and_line(self):
        x = np.random.randn(50)
        y = x * 2 + 1
        fig, ax = plot_scatter_with_regression(x, y)
        # Should have at least one PathCollection (scatter) and one Line2D (regression)
        collections = ax.collections
        lines = ax.get_lines()
        assert len(collections) >= 1, "No scatter plot found"
        assert len(lines) >= 1, "No regression line found"


# ---------------------------------------------------------------------------
# plot_correlation_heatmap
# ---------------------------------------------------------------------------

class TestPlotCorrelationHeatmap:
    def _make_df(self):
        rng = np.random.RandomState(42)
        n = 100
        a = rng.randn(n)
        b = a * 2 + rng.randn(n) * 0.5
        c = rng.randn(n)
        return pd.DataFrame({"feat_a": a, "feat_b": b, "feat_c": c})

    def test_returns_figure_and_axes(self):
        result = plot_correlation_heatmap(self._make_df())
        assert isinstance(result, tuple) and len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_title_is_set(self):
        fig, ax = plot_correlation_heatmap(self._make_df(), title="My Heatmap")
        assert ax.get_title() == "My Heatmap"

    def test_figure_size(self):
        fig, ax = plot_correlation_heatmap(self._make_df())
        width, height = fig.get_size_inches()
        assert abs(width - 10) < 0.1 and abs(height - 8) < 0.1

    def test_has_annotations(self):
        """Heatmap should have text annotations showing correlation values."""
        fig, ax = plot_correlation_heatmap(self._make_df())
        texts = ax.texts
        assert len(texts) > 0, "Heatmap should have text annotations"

    def test_annotations_are_numeric(self):
        fig, ax = plot_correlation_heatmap(self._make_df())
        for text_obj in ax.texts:
            val = text_obj.get_text()
            try:
                float(val)
            except ValueError:
                pytest.fail(f"Annotation '{val}' is not a number")


# ---------------------------------------------------------------------------
# plot_training_curves
# ---------------------------------------------------------------------------

class TestPlotTrainingCurves:
    def _make_curves(self, n_epochs=20):
        epochs = np.arange(1, n_epochs + 1)
        train = 1.0 / (1 + 0.1 * epochs) + np.random.randn(n_epochs) * 0.01
        val = 1.0 / (1 + 0.08 * epochs) + np.random.randn(n_epochs) * 0.02
        return train, val

    def test_returns_figure_and_axes(self):
        train, val = self._make_curves()
        result = plot_training_curves(train, val)
        assert isinstance(result, tuple) and len(result) == 2
        fig, ax = result
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_title_is_set(self):
        train, val = self._make_curves()
        fig, ax = plot_training_curves(train, val, title="Loss Over Time")
        assert ax.get_title() == "Loss Over Time"

    def test_xlabel_and_ylabel(self):
        train, val = self._make_curves()
        fig, ax = plot_training_curves(train, val, xlabel="Step", ylabel="MSE")
        assert ax.get_xlabel() == "Step"
        assert ax.get_ylabel() == "MSE"

    def test_has_two_lines(self):
        train, val = self._make_curves()
        fig, ax = plot_training_curves(train, val)
        lines = ax.get_lines()
        assert len(lines) >= 2, "Should have at least 2 lines (train and validation)"

    def test_has_legend(self):
        train, val = self._make_curves()
        fig, ax = plot_training_curves(train, val)
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert any("train" in l.lower() for l in labels), f"No 'Train' in legend: {labels}"
        assert any("val" in l.lower() for l in labels), f"No 'Validation' in legend: {labels}"

    def test_has_grid(self):
        train, val = self._make_curves()
        fig, ax = plot_training_curves(train, val)
        # Check that grid is enabled on at least one axis
        assert ax.xaxis.get_gridlines()[0].get_visible() or ax.yaxis.get_gridlines()[0].get_visible()

    def test_x_axis_starts_at_one(self):
        train, val = self._make_curves(n_epochs=10)
        fig, ax = plot_training_curves(train, val)
        lines = ax.get_lines()
        # The first line's x-data should start at 1
        first_line_x = lines[0].get_xdata()
        assert first_line_x[0] == 1, f"X-axis should start at 1, got {first_line_x[0]}"
