# Module 04: Visualization

> **Time estimate:** 2--3 hours
> **Prerequisites:** Modules 02 and 03 complete
> **Math prep:** [StatQuest: Histograms](https://www.youtube.com/watch?v=qBigTkBLU6g) and [StatQuest: Correlation](https://www.youtube.com/watch?v=xZ_z8KWkhXE)

## Why Visualization Matters for ML

You can't debug what you can't see. Before training any model, you need to *look* at your data: Are the features normally distributed? Are there outliers? Which features correlate with the target? During training, you need to *watch* the loss curves: Is the model converging? Is it overfitting?

Visualization isn't a nice-to-have. It's a core debugging tool.

## Concepts

### 1. The Matplotlib Object Model

Matplotlib has two interfaces:
- **pyplot (plt):** Stateful, MATLAB-style. Quick but brittle for complex layouts.
- **Object-oriented (Figure/Axes):** Explicit, composable, what you should use.

> **Go parallel:** The pyplot API is like using global state---quick for scripts, painful for anything complex. The OO API is like passing an explicit `context` or `writer` parameter. You always know which plot you're drawing on.

```python
# The right way: OO interface
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, y)
ax.set_title("My Plot")
ax.set_xlabel("X Axis")
```

**Key objects:**
- `Figure` — the entire image/canvas
- `Axes` — a single plot within the figure (confusingly, not the axis lines)
- `Axis` — the actual x-axis or y-axis (tick marks, labels, limits)

### 2. Distribution Plots

Histograms show how values are distributed. This is your first look at any feature.

**What to look for:**
- **Shape:** Normal (bell curve)? Skewed? Bimodal?
- **Spread:** Tight or wide? What are the min/max?
- **Outliers:** Any bars far from the main mass?

If a feature is highly skewed, you might want to log-transform it before feeding it to a model.

> **Math resource:** [3Blue1Brown: But what is a convolution?](https://www.youtube.com/watch?v=KuXjwB4LzSA) --- not directly about histograms, but helps build intuition about distributions and smoothing.

### 3. Scatter Plots with Regression Lines

Scatter plots reveal the relationship between two continuous variables. Adding a regression line shows the trend.

**What to look for:**
- **Direction:** Positive or negative relationship?
- **Strength:** Tight cluster around the line or wide scatter?
- **Non-linearity:** Does the relationship curve? A straight line might not capture it.
- **Heteroscedasticity:** Does the spread change across the x-axis? This violates linear regression assumptions.

### 4. Correlation Heatmaps

A correlation matrix shows pairwise Pearson correlations between all numeric features. The heatmap visualizes this matrix with color intensity.

**What to look for:**
- **Strong positive correlation** (near +1): Features move together. One might be redundant.
- **Strong negative correlation** (near -1): Features move oppositely.
- **Near zero:** No linear relationship (but could still have a non-linear one).
- **Target correlations:** Which features correlate most with the target variable? Those are your best candidates for a first model.

> **Math resource:** [StatQuest: Covariance and Correlation](https://www.youtube.com/watch?v=xZ_z8KWkhXE)

### 5. Training Curves

During model training, you plot loss (and/or accuracy) over epochs for both training and validation sets.

**What to look for:**
- **Converging:** Both curves going down and flattening? Good.
- **Overfitting:** Training loss keeps dropping but validation loss goes up? The model is memorizing, not learning.
- **Underfitting:** Both curves plateau at a high loss? The model isn't complex enough.
- **Learning rate issues:** Loss is spiky or diverging? Learning rate is too high.

You'll produce these plots constantly once you start training models in later phases.

## Exercises

Implement the stubs in `exercises.py`. Run `pytest tests/test_exercises.py -v`.

| Function | What It Practices |
|----------|------------------|
| `plot_distribution` | Histograms, KDE overlay, axis labels, title |
| `plot_scatter_with_regression` | Scatter plots, line fitting, legends |
| `plot_correlation_heatmap` | Seaborn heatmap, annotation, diverging colormaps |
| `plot_training_curves` | Multi-line plots, train/val distinction, legends |

**Note on testing:** The tests verify that your functions return proper `Figure` and `Axes` objects with the correct titles, labels, and structural properties. They don't check pixel-level appearance---that's your job to eyeball.

## Style Notes

- **Always use the OO interface** (`fig, ax = plt.subplots()`). Never use bare `plt.plot()` in anything beyond a one-off script.
- **Always label axes and add a title.** An unlabeled plot is useless to anyone (including future-you).
- **Close figures after saving** with `plt.close(fig)` to avoid memory leaks in notebooks and long-running scripts.
- **Use consistent colormaps.** `viridis` for sequential data, `RdBu` or `coolwarm` for diverging (correlation). Never use `jet`---it's perceptually non-uniform and misleading.
- **Set `figsize` explicitly.** Default matplotlib figures are too small for presentations and too large for inline notebook display. `(8, 5)` or `(10, 6)` are good starting points.
- **Use seaborn for statistical plots** when available. `sns.heatmap` with `annot=True` is much easier than building a heatmap from scratch in matplotlib.
