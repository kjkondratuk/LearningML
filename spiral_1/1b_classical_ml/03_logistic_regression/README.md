# Module 03: Logistic Regression

## Overview

Logistic regression is linear regression's cousin for **classification**. Despite
the name, it is not a regression algorithm — it predicts probabilities of discrete
classes. The key insight: take a linear combination of features, then squeeze it
through a sigmoid function to get a probability.

## Core Concepts

### The Sigmoid Function

    sigmoid(z) = 1 / (1 + exp(-z))

This S-shaped curve maps any real number to the range (0, 1). Large positive
values get mapped close to 1, large negative values close to 0, and 0 maps to 0.5.

**Go parallel:** Think of sigmoid as a type conversion — you have a `float64` that
could be anything, and you need to coerce it into a valid probability. Sigmoid is
the "adapter" that does this smoothly and differentiably.

### From Probability to Prediction

    p = sigmoid(X @ w + b)
    y_hat = 1 if p >= 0.5 else 0

The threshold 0.5 is the default, but you can tune it. Module 07 (Model Evaluation)
explores why you might want a different threshold.

### Log Loss (Binary Cross-Entropy)

Why not use MSE for classification? Because MSE creates a non-convex loss landscape
with many local minima when combined with the sigmoid. Log loss is convex and has a
single global minimum:

    L = -(1/n) * sum(y_i * log(p_i) + (1 - y_i) * log(1 - p_i))

Intuition: if the true label is 1 and you predict p=0.01, the term `log(0.01)` is
a huge penalty. If you predict p=0.99, the penalty is tiny. The loss function
heavily punishes confident wrong answers.

### Gradients

For log loss with sigmoid:

    dL/dw = (1/n) * X^T @ (p - y)
    dL/db = (1/n) * sum(p - y)

These look nearly identical to the linear regression gradients. The sigmoid handles
the nonlinearity.

Resources:
- [StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
- [Scikit-learn LogisticRegression docs](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [3Blue1Brown: Neural Networks (Ch.1)](https://www.youtube.com/watch?v=aircAruvnKk) — sigmoid explanation applies here

## Exercises

Implement in `exercises.py`:

1. `sigmoid` — The sigmoid activation function.
2. `predict_proba` — Compute class probabilities using sigmoid(X @ w + b).
3. `predict_class` — Threshold probabilities to get binary predictions.
4. `compute_log_loss` — Binary cross-entropy loss.
5. `train_logistic_regression` — Full training loop with gradient descent.
6. `sklearn_logistic_regression` — Fit and predict using sklearn.

## Do It the Wrong Way (on purpose)

After your implementation works, try using MSE loss instead of log loss for
training a logistic regression. You will observe:

1. Training is slower and less stable.
2. The loss surface has flat regions where gradients vanish.
3. The final model may settle in a poor local minimum.

This is why the choice of loss function matters — it is not just an academic detail.

## Style Notes

- Clip probabilities to [1e-15, 1 - 1e-15] inside log loss to avoid log(0).
- The sigmoid should handle large positive and negative inputs without overflow.
  NumPy's `np.exp` will overflow for large positive z in the denominator; consider
  using `np.clip` or the numerically stable form.
- Return probabilities as 1D arrays of shape (n_samples,).
