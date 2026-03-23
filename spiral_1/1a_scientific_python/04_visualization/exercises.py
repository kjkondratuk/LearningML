"""Module 04: Visualization — Exercise Stubs

Implement each function below until all tests in tests/test_exercises.py pass.
Each function should create a matplotlib Figure and Axes, draw the requested plot,
and return the (fig, ax) tuple.
"""

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray

# Use non-interactive backend for testing (no GUI windows)
matplotlib.use("Agg")


def plot_distribution(
    data: NDArray[np.floating],
    title: str = "Distribution",
    xlabel: str = "Value",
    ylabel: str = "Frequency",
    bins: int = 30,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a histogram of the data with a KDE (kernel density estimate) overlay.

    Requirements:
    - Create a figure with figsize=(8, 5).
    - Plot a histogram with the specified number of bins and alpha=0.7.
    - Overlay a KDE curve (use scipy.stats.gaussian_kde or seaborn).
    - Set the title, xlabel, and ylabel as specified.
    - Add a legend with entries "Histogram" and "KDE".

    Args:
        data: 1D array of numeric values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        bins: Number of histogram bins.

    Returns:
        Tuple of (Figure, Axes).
    """
    raise NotImplementedError


def plot_scatter_with_regression(
    x: NDArray[np.floating],
    y: NDArray[np.floating],
    title: str = "Scatter with Regression",
    xlabel: str = "X",
    ylabel: str = "Y",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a scatter plot of x vs y with a linear regression line.

    Requirements:
    - Create a figure with figsize=(8, 5).
    - Plot the data points as a scatter plot with alpha=0.5.
    - Fit a linear regression line (use np.polyfit degree 1) and plot it in red.
    - Set the title, xlabel, and ylabel.
    - Add a legend with entries "Data" and "Regression".

    Args:
        x: 1D array of x values.
        y: 1D array of y values.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Tuple of (Figure, Axes).
    """
    raise NotImplementedError


def plot_correlation_heatmap(
    df: pd.DataFrame,
    title: str = "Correlation Heatmap",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a correlation heatmap of all numeric columns in the DataFrame.

    Requirements:
    - Create a figure with figsize=(10, 8).
    - Compute the Pearson correlation matrix of numeric columns.
    - Use seaborn's heatmap with annot=True, fmt=".2f", and a diverging colormap
      (cmap="coolwarm") centered at 0 (vmin=-1, vmax=1).
    - Set the title.

    Args:
        df: DataFrame with numeric columns.
        title: Plot title.

    Returns:
        Tuple of (Figure, Axes).
    """
    raise NotImplementedError


def plot_training_curves(
    train_losses: NDArray[np.floating],
    val_losses: NDArray[np.floating],
    title: str = "Training Curves",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot training and validation loss curves over epochs.

    Requirements:
    - Create a figure with figsize=(8, 5).
    - Plot train_losses as a solid blue line labeled "Train".
    - Plot val_losses as a dashed orange line labeled "Validation".
    - The x-axis represents epochs (1-indexed: 1, 2, 3, ...).
    - Set the title, xlabel, and ylabel.
    - Add a legend.
    - Add a grid with alpha=0.3.

    Args:
        train_losses: 1D array of training loss values per epoch.
        val_losses: 1D array of validation loss values per epoch.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.

    Returns:
        Tuple of (Figure, Axes).
    """
    raise NotImplementedError
