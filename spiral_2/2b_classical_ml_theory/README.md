# Phase 2B: Classical ML (Theory Pass)

> **Goal:** Derive and implement classical ML algorithms from scratch. By the end of
> this phase you should be able to open Bishop's PRML to any chapter on classical
> methods and follow the derivations.

## Spiral 2 Principles

- **No scikit-learn in implementations.** Only NumPy and your Phase 2A mathlib.
  Scikit-learn appears as a reference oracle in tests.
- **"Naive then Derive" pedagogy.** Every module includes a challenge where you
  build the naive version first, discover its limitations, then derive the
  principled solution.
- **MLE/MAP/Bayesian progression.** Modules follow a common arc: start with MLE,
  add a prior to get MAP (regularization), then go full Bayesian where tractable.

## Module Index

| # | Module | Core Idea | Key Deliverable |
|---|--------|-----------|-----------------|
| 01 | [Linear Regression](01_linear_regression/) | OLS, MLE, MAP, Bayesian | Bayesian regression with uncertainty bands |
| 02 | [Logistic Regression](02_logistic_regression/) | Sigmoid, Newton-Raphson, softmax | Multi-class MNIST classifier |
| 03 | [Regularization](03_regularization/) | L1 sparsity, L2 shrinkage, elastic net | Regularization path visualization |
| 04 | [Bias-Variance](04_bias_variance/) | The decomposition, double descent | Classic U-shaped curve + double descent |
| 05 | [SVMs](05_svms/) | Margin, duality, kernel trick | Kernel SVM on non-linear data |
| 06 | [Trees & Ensembles](06_decision_trees_and_ensembles/) | Gini, bagging, boosting | Random forest from scratch |
| 07 | [Dimensionality Reduction](07_dimensionality_reduction/) | PCA, kernel PCA, t-SNE | PCA + t-SNE on Fashion-MNIST |
| 08 | [Capstone: ML Library](08_capstone_ml_from_scratch/) | Full sklearn-like API from scratch | Complete pipeline on real data |

## Estimated Timeline

6--8 weeks. Seven algorithm modules plus a capstone.

## Prerequisites

- Phase 2A completed (mathlib available)
- Comfort with matrix calculus, MLE/MAP, gradient-based optimization
