# Module 07: Model Evaluation

## Overview

A model is only as good as how you measure it. This module covers the evaluation
tools that separate useful models from misleading ones. The central lesson:
**accuracy is not enough**.

## Core Concepts

### Confusion Matrix

For binary classification, every prediction falls into one of four boxes:

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| Actually Positive | True Positive (TP) | False Negative (FN) |
| Actually Negative | False Positive (FP) | True Negative (TN) |

**Go parallel:** Think of a confusion matrix like a unit test result matrix. TP =
test passes, code is correct. FP = test passes, but code is wrong (false sense of
security). FN = test fails, but code is actually correct (flaky test). TN = test
fails because code is wrong (correct failure).

### Precision, Recall, F1

- **Precision** = TP / (TP + FP) — "Of everything I called positive, how many
  actually were?"
- **Recall** = TP / (TP + FN) — "Of everything that was actually positive, how
  many did I catch?"
- **F1** = 2 * precision * recall / (precision + recall) — Harmonic mean of
  precision and recall.

When to care about which:
- **Spam filter:** High precision matters (do not put real email in spam).
- **Cancer screening:** High recall matters (do not miss a case).

### ROC Curve

The ROC curve plots True Positive Rate (recall) vs. False Positive Rate at
different classification thresholds. The area under the curve (AUC) summarizes
model quality: 1.0 is perfect, 0.5 is random guessing.

### Cross-Validation

Instead of a single train/test split, K-fold cross-validation splits data into K
folds, trains on K-1, tests on 1, and rotates. This gives K scores, and their mean
is a more reliable estimate of model performance.

**Go parallel:** Cross-validation is like running your test suite with K different
random seeds and averaging the results. A single run might get lucky; K runs give
you confidence.

### Learning Curves

Plot training and validation scores as a function of training set size. This
diagnoses whether you need more data, a more complex model, or more regularization.

Resources:
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Google ML Crash Course: Classification](https://developers.google.com/machine-learning/crash-course/classification)
- [StatQuest: ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJemM)

## Exercises

Implement in `exercises.py`:

1. `confusion_matrix_manual` — Build a 2x2 confusion matrix from scratch.
2. `precision_recall_f1` — Compute precision, recall, F1 from y_true and y_pred.
3. `roc_points` — Compute (FPR, TPR) pairs at multiple thresholds.
4. `cross_validate_manual` — Implement K-fold cross-validation from scratch.
5. `learning_curve_data` — Compute train/test scores at increasing training sizes.

## Do It the Wrong Way (on purpose)

Create an imbalanced dataset (95% class 0, 5% class 1). Build a model that
always predicts class 0. Observe:

1. Accuracy is 95%. Looks great.
2. Recall for class 1 is 0%. The model is useless for the minority class.
3. The confusion matrix makes this obvious. Accuracy hides it.

This is why you need precision, recall, and the confusion matrix, not just accuracy.

## Style Notes

- `confusion_matrix_manual` should return a 2x2 NumPy array with layout:
  [[TN, FP], [FN, TP]].
- `roc_points` should return sorted (FPR, TPR) arrays suitable for plotting.
- For `cross_validate_manual`, do not use sklearn's cross_val_score. Implement
  the fold logic yourself using a simple classifier you train and evaluate.
