# 04 -- Optimization Intuition

## Learning Objectives

- Determine whether a 1D function is convex on an interval
- Implement gradient descent with momentum
- Compare vanilla SGD, SGD+momentum, and Adam
- Implement a learning rate schedule
- Visualize optimization trajectories on a 2D loss surface

## Why This Matters for ML

Training any ML model is an optimization problem: find parameters that minimize
the loss. The choice of optimizer, learning rate, and schedule can mean the
difference between a model that converges in minutes and one that never learns.

### Go Parallel

Gradient descent with momentum in Go:

```go
velocity := 0.0
for i := 0; i < maxIter; i++ {
    grad := derivative(f, x)
    velocity = momentum * velocity - lr * grad
    x += velocity
}
```

Momentum is like a ball rolling downhill -- it builds up speed in consistent
directions and dampens oscillations. Adam adds per-parameter adaptive learning
rates on top of momentum.

## Key Concepts

### 1. Convexity

A function is convex if the line segment between any two points on its graph
lies above the graph. Convex functions have a single global minimum, making
optimization easy.

> **Math Callout:** A twice-differentiable function is convex if and only if
> its second derivative is non-negative everywhere. MSE loss for linear
> regression is convex. Neural network loss functions are generally non-convex,
> which is why initialization and learning rate matter.

### 2. Momentum

Standard gradient descent can oscillate in narrow valleys. Momentum smooths
the trajectory by accumulating a velocity term:

```
v_t = beta * v_{t-1} + (1 - beta) * grad
x_t = x_{t-1} - lr * v_t
```

### 3. Adam Optimizer

Combines momentum (first moment) with RMSProp (second moment, which adapts
the learning rate per parameter). Default choice for most deep learning.

### 4. Learning Rate Schedules

Start with a higher learning rate (explore broadly) and decay over time
(fine-tune). Common schedules:
- Step decay: halve LR every N epochs
- Cosine annealing: smooth decay following a cosine curve
- Warmup: start small, ramp up, then decay

### 5. 2D Loss Surfaces

Visualizing how different optimizers navigate a 2D loss surface (e.g.,
the Rosenbrock function or Beale function) builds powerful intuition.

## Mini-Project: Visualize Optimization Landscape

Plot optimization trajectories of SGD, SGD+momentum, and Adam on a 2D
function like the Rosenbrock function. Show how momentum helps escape
oscillations and Adam adapts step sizes.

## Style Notes

- Return trajectory histories (list of (x, y, loss)) so they can be plotted.
- Use small, deterministic test functions for unit tests.
- Keep optimizer implementations functional: take current state, return new state.

## Exercises

See [exercises.py](exercises.py) for 5 stubs and
[mini_project.py](mini_project.py) for the visualization project. Tests in
`tests/`.
