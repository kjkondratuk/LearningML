# 02 -- Calculus for ML

## Learning Objectives

- Approximate derivatives numerically (finite differences)
- Compute partial derivatives of multivariate functions
- Derive and implement the gradient of MSE loss
- Apply the chain rule to composed functions
- Implement gradient descent in 1D and 2D

## Why This Matters for ML

Training a model means minimizing a loss function. To minimize, you need
the gradient -- the direction of steepest ascent. Calculus gives you the
gradient; gradient descent uses it to update parameters.

### Go Parallel

In Go, numerical differentiation is straightforward:

```go
func derivative(f func(float64) float64, x, h float64) float64 {
    return (f(x+h) - f(x-h)) / (2 * h)
}
```

Gradient descent is a loop:

```go
for i := 0; i < maxIter; i++ {
    grad := derivative(f, x, 1e-7)
    x -= lr * grad
}
```

That is literally all of calculus you need to train a neural network (PyTorch
computes the derivatives automatically, but you should understand what it does).

## Key Concepts

### 1. Numerical Derivative

```
f'(x) ~ (f(x+h) - f(x-h)) / (2h)
```

This is the **central difference** approximation. It is O(h^2) accurate,
meaning the error shrinks quadratically as h decreases.

### 2. Partial Derivatives

For a function `f(x, y)`, the partial derivative with respect to x is
`df/dx` -- differentiate treating y as constant. The gradient is the vector
of all partial derivatives: `grad f = [df/dx, df/dy]`.

### 3. Gradient of MSE

For MSE loss `L = (1/n) * sum((y_pred - y_true)^2)` with `y_pred = X @ w`:
```
dL/dw = (2/n) * X.T @ (X @ w - y_true)
```

> **Math Callout:** This is the gradient that linear regression uses. Setting
> it to zero gives the normal equation `w = (X.T @ X)^{-1} @ X.T @ y`.
> See 1B-02 for the ML context.

### 4. Chain Rule

If `y = f(g(x))`, then `dy/dx = f'(g(x)) * g'(x)`.

This is the foundation of backpropagation: each layer in a neural network
applies the chain rule to pass gradients backward. See 1C-02 for the
implementation.

### 5. Gradient Descent

```
x_new = x_old - lr * gradient(x_old)
```

Repeat until convergence. The learning rate `lr` controls step size.

## Style Notes

- Use `h = 1e-7` for numerical derivatives (smaller values cause floating
  point issues).
- Always test analytical gradients against numerical ones (gradient checking).

## Exercises

See [exercises.py](exercises.py) for 6 stubs. Tests in
`tests/test_exercises.py`.
