# Phase 1D: Math Refresher (Reference Module)

## Goal

This is a **reference module**, not a sequential course. Dip into it whenever
a concept in Phase 1B (Classical ML) or 1C (Neural Networks) feels shaky.
Each section gives you just enough math to understand the ML algorithms, with
exercises that connect directly to code.

## Cross-Reference Table

Use this table to find the math behind specific ML concepts.

| Math Concept | Refresher Module | Where It Appears in 1B/1C |
|---|---|---|
| Dot product | 1D-01 | 1B-02 (linear regression weights), 1C-01 (perceptron) |
| Matrix multiply | 1D-01 | 1C-02 (forward pass), 1C-05 (convolution as matmul) |
| Linear transforms | 1D-01 | 1C-02 (layers are linear transforms + activation) |
| Covariance matrix | 1D-01 | 1B-02 (feature correlations), PCA |
| Eigendecomposition / PCA | 1D-01 | 1B feature engineering, dimensionality reduction |
| Numerical derivatives | 1D-02 | 1C-02 (gradient checking) |
| Partial derivatives | 1D-02 | 1C-02 (backpropagation) |
| Chain rule | 1D-02 | 1C-02 (backprop is the chain rule), 1C-06 (BPTT) |
| Gradient of MSE | 1D-02 | 1B-02 (linear regression gradient descent) |
| Gradient descent | 1D-02, 1D-04 | 1B-02, 1B-03, 1C-02, 1C-04 (training loops) |
| Probability basics | 1D-03 | 1B-03 (logistic regression), 1B-04 (Naive Bayes) |
| Bayes' theorem | 1D-03 | 1B-04 (decision trees / Naive Bayes), 1C-01 (softmax) |
| Gaussian PDF | 1D-03 | 1B-02 (error assumptions), 1B-04 (Naive Bayes) |
| Naive Bayes | 1D-03 | 1B-04 (ensemble methods baseline) |
| Convexity | 1D-04 | 1B-02 (MSE is convex), 1C loss landscapes |
| Momentum | 1D-04 | 1C-04 (SGD with momentum, Adam) |
| Learning rate schedules | 1D-04 | 1C-04, 1C-07 (fine-tuning LR) |
| Optimizer comparison | 1D-04 | 1C-04 (Adam vs SGD) |

## Module Index

| # | Module | Key Ideas | Exercises |
|---|--------|-----------|-----------|
| 01 | [Linear Algebra Essentials](01_linear_algebra_essentials/) | Dot product, matmul, transforms, eigenvalues, PCA | 6 stubs |
| 02 | [Calculus for ML](02_calculus_for_ml/) | Derivatives, partials, chain rule, gradient descent | 6 stubs |
| 03 | [Probability & Statistics](03_probability_and_statistics/) | Probability, Bayes, Gaussian, Naive Bayes | 6 stubs |
| 04 | [Optimization Intuition](04_optimization_intuition/) | Convexity, momentum, LR schedules, optimizer comparison | 5 stubs + mini-project |

## How to Use This Module

1. **Do not try to read it front-to-back.** Use the cross-reference table
   above to find what you need.
2. **When a 1B or 1C exercise feels confusing**, come here and do the matching
   exercises. The math will make the ML click.
3. **The "wrong way" challenges** show you common math mistakes and why they
   matter in practice.
4. **Module 04 (Optimization)** is the most directly applicable -- do it
   alongside or right after 1C-04 (Training Loop).

## Running Tests

```bash
# Run all 1D tests
python -m pytest spiral_1/1d_math_refresher/ -v

# Run a single module
python -m pytest spiral_1/1d_math_refresher/01_linear_algebra_essentials/tests/ -v
```
