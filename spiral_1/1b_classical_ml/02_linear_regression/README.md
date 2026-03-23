# Module 02: Linear Regression

## Overview

Linear regression is the "Hello, World" of machine learning. It fits a straight
line (or hyperplane) through your data by minimizing the squared error between
predictions and actual values.

You will implement it two ways: manually with gradient descent, and then using
scikit-learn. Doing both builds the intuition that sklearn is not magic — it is
running the same math you just wrote, only faster and with more edge-case handling.

## Core Concepts

### The Model

A linear model predicts:

    y_hat = X @ w + b

where `w` is a weight vector (one weight per feature) and `b` is a bias (intercept).

**Go parallel:** Think of `w` as a config struct's fields and `b` as a default value.
The model is a simple function: take the inputs, multiply by weights, add the bias.
No hidden state, no goroutines — just matrix multiplication.

### The Loss Function (MSE)

To measure how wrong your predictions are:

    L = (1/n) * sum((y_hat_i - y_i)^2)

The goal of training is to find `w` and `b` that minimize L.

### Gradient Descent

Gradient descent is an iterative optimizer. At each step:

1. Compute predictions with current weights.
2. Compute the loss.
3. Compute the gradient (partial derivatives of loss with respect to each weight).
4. Update weights: `w = w - learning_rate * gradient`.

The learning rate controls step size. Too large and you overshoot the minimum.
Too small and training takes forever.

**Go parallel:** This is like a feedback loop in a control system. You observe the
error, compute a correction, apply it, and repeat. The learning rate is your
damping factor. Set it wrong and the system oscillates (too high) or barely
moves (too low).

### Gradient Formulas

For MSE loss with predictions y_hat = X @ w + b:

    dL/dw = (2/n) * X^T @ (y_hat - y)
    dL/db = (2/n) * sum(y_hat - y)

Resources:
- [3Blue1Brown: Gradient Descent](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- [Andrew Ng: Linear Regression (Coursera)](https://www.coursera.org/learn/machine-learning)
- [Scikit-learn LinearRegression docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

## Exercises

Implement in `exercises.py`:

1. `predict_linear` — Forward pass: X @ w + b.
2. `compute_loss_mse` — MSE between predictions and targets.
3. `gradient_descent_step` — One step of gradient descent for w and b.
4. `train_linear_regression` — Full training loop using gradient descent.
5. `sklearn_linear_regression` — Fit and predict using sklearn.

## Mini-Project

In `mini_project.py`:

`predict_housing_prices` — Build a linear regression pipeline on the California
Housing dataset (or a synthetic stand-in). Split, train, evaluate, return the
results.

## Do It the Wrong Way (on purpose)

After your basic implementation works:

1. **Learning rate too high (e.g., 10.0):** Watch the loss explode to infinity
   or NaN. This is gradient descent diverging.
2. **Learning rate too low (e.g., 1e-10):** Watch the loss barely change after
   1000 iterations. The model learns nothing useful.
3. **No feature scaling:** Train on features with wildly different scales (e.g.,
   one column in [0, 1] and another in [0, 1000000]). Observe how gradient
   descent struggles or fails.

These failures are more instructive than the successes.

## Style Notes

- Weights `w` should be a 1D NumPy array of shape (n_features,).
- Bias `b` should be a scalar float.
- Use `np.dot` or `@` for matrix multiplication — never loop over samples.
- Return types should always be NumPy arrays or plain floats, not sklearn objects.
