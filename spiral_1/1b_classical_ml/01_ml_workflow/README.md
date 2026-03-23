# Module 01: The ML Workflow

## Overview

Before touching any algorithm, you need to understand the **workflow** that every
ML project follows. This module covers the plumbing that prevents you from fooling
yourself with bad results.

Think of it like writing tests before code in Go: the evaluation harness comes
first, the model comes second.

## Core Concepts

### Train/Test Split

You never evaluate a model on the same data it learned from. This is the single
most important rule in ML, and the one most often violated by beginners.

**Go parallel:** Imagine writing a Go test where the test cases are the exact same
inputs your function was hardcoded to handle. The test passes, but the function is
useless on new input. That is what evaluating on training data looks like.

The standard split is 80/20 or 70/30 (train/test). You shuffle first to avoid
ordering bias.

### Evaluation Metrics

Different problems need different scorecards:

- **Accuracy** = correct predictions / total predictions. Simple but misleading on
  imbalanced data (Module 07 goes deeper).
- **MSE (Mean Squared Error)** = average of (prediction - actual)^2. Penalizes
  large errors heavily. Used for regression.
- **MAE (Mean Absolute Error)** = average of |prediction - actual|. More robust
  to outliers than MSE.

### Data Leakage

Data leakage is when information from the test set bleeds into your training
process. It is the ML equivalent of a race condition: your results look great
but they are wrong, and the bug is subtle.

Common sources of leakage:
- Scaling/normalizing using the full dataset (including test) before splitting
- Feature engineering that uses future data
- Duplicate rows spanning train and test sets

**Go parallel:** This is like a goroutine reading from a channel it should not
have access to. The program runs, the output looks right, but the logic is
fundamentally broken.

## Math Callouts

**MSE formula:**

MSE = (1/n) * sum((y_pred_i - y_true_i)^2) for i = 1..n

**MAE formula:**

MAE = (1/n) * sum(|y_pred_i - y_true_i|) for i = 1..n

These are straightforward — just averages of errors. The squared version (MSE)
penalizes big misses more.

Resources:
- [Scikit-learn model evaluation docs](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Google ML Crash Course: Training and Test Sets](https://developers.google.com/machine-learning/crash-course/training-and-test-sets)

## Exercises

Implement in `exercises.py`, drive with `pytest tests/`:

1. `train_test_split_manual` — Split data into train and test sets with shuffling.
2. `compute_accuracy` — Fraction of correct classification predictions.
3. `compute_mse` — Mean squared error for regression.
4. `compute_mae` — Mean absolute error for regression.
5. `detect_data_leakage` — Check for row overlap between train and test sets.

## Do It the Wrong Way (on purpose)

After your tests pass, try this:

1. Train a simple model (even just "predict the mean") on the full dataset.
2. Evaluate it on the same data it was trained on.
3. Compare that score to evaluating on a held-out test set.

You will see the training-set score is suspiciously good. This is overfitting,
and it is the reason the train/test split exists.

## Style Notes

- Use NumPy arrays, not Python lists, for all data.
- Functions should work with 1D arrays (for targets) and 2D arrays (for features).
- Use `np.random.RandomState` (not `np.random.seed`) for reproducible shuffling.
  This is the equivalent of injecting a deterministic RNG in Go tests.
