# Module 05: Support Vector Machines (SVM)

## Overview

SVMs find the decision boundary that maximizes the **margin** — the distance
between the boundary and the nearest data points (support vectors). This module
is sklearn-focused: you will not implement SVM from scratch, but you will build
strong intuition for what kernels do and how to tune them.

## Core Concepts

### The Maximum Margin Idea

Given two linearly separable classes, there are infinitely many lines that
separate them. SVM picks the one with the widest "gap" between the classes.
The wider the margin, the more confident the classifier is and the better it
tends to generalize.

**Go parallel:** Think of the margin as a buffer zone. In Go, you might design
an API with strict input validation (wide margin) vs. one that barely rejects
invalid input (narrow margin). The strict API handles edge cases better.

### Kernels and the Kernel Trick

When data is not linearly separable, SVM maps it to a higher-dimensional space
where a linear boundary exists. The kernel trick does this mapping implicitly
(without computing the actual high-dimensional coordinates), which keeps it fast.

Common kernels:
- **Linear:** No transformation. Works when data is (nearly) linearly separable.
- **RBF (Radial Basis Function):** Maps to infinite-dimensional space. Very
  flexible. The default in sklearn.
- **Polynomial:** Maps to a finite higher-dimensional space.

### The C Parameter

C controls the tradeoff between a wide margin and correctly classifying training
points:
- **High C:** Narrow margin, fewer misclassifications on training data. Risk of
  overfitting.
- **Low C:** Wide margin, more training errors allowed. More regularized.

### The Gamma Parameter (RBF kernel)

Gamma controls how far the influence of a single training example reaches:
- **High gamma:** Each point only influences nearby points. Complex, wiggly
  boundaries. Risk of overfitting.
- **Low gamma:** Each point influences far-away points. Smooth boundaries.
  Risk of underfitting.

Resources:
- [StatQuest: SVM](https://www.youtube.com/watch?v=efR1C6CvhmE)
- [Scikit-learn SVM docs](https://scikit-learn.org/stable/modules/svm.html)
- [Visual SVM demo](https://cs.stanford.edu/people/karpathy/svmjs/demo/)

## Exercises

Implement in `exercises.py`:

1. `sklearn_svm_linear` — Train a linear SVM and return predictions.
2. `sklearn_svm_rbf` — Train an RBF SVM and return predictions.
3. `compare_kernels` — Train SVMs with different kernels, return accuracy dict.
4. `visualize_decision_boundary` — Generate a meshgrid and predict on it for
   visualization (return the mesh predictions, not a plot).
5. `grid_search_svm` — Use GridSearchCV to find optimal C and gamma for RBF SVM.

## Style Notes

- All functions use sklearn's SVC, not your own implementation.
- For `visualize_decision_boundary`, return the meshgrid coordinates and
  predictions so the caller can plot them. Do not call matplotlib inside the
  function.
- Use `sklearn.model_selection.GridSearchCV` for the grid search exercise.
- Set `random_state` where applicable for reproducibility.
