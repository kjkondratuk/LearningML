# 02 -- Neural Net from Scratch

## Learning Objectives

- Initialize weights with proper random scaling
- Implement a full forward pass (matrix multiply + activation)
- Derive and implement backpropagation for a two-layer network
- Verify your gradients with numerical gradient checking
- Train a network to classify the "spirals" dataset

## The Big Idea

A neural network is a **composition of simple functions**: linear transforms
followed by nonlinearities, stacked repeatedly. Training means finding weights
that minimize a loss function, using the chain rule to compute how each weight
contributes to the error.

### Go Parallel

In Go terms, a two-layer network is:

```go
func forward(x []float64, W1, W2 [][]float64, b1, b2 []float64) []float64 {
    hidden := relu(add(matmul(W1, x), b1))
    output := softmax(add(matmul(W2, hidden), b2))
    return output
}
```

Backpropagation is computing `d(loss)/d(W1)` and `d(loss)/d(W2)` so you know
which direction to nudge each weight. The chain rule is your friend.

## Key Concepts

### 1. Weight Initialization

Random, but scaled. A common approach (He initialization):

```
W = np.random.randn(out_dim, in_dim) * sqrt(2.0 / in_dim)
```

> **Math Callout:** If weights are too large, activations explode. Too small,
> they vanish. He initialization keeps the variance of activations roughly
> constant across layers. See 1D-01 for the linear algebra behind this.

### 2. Forward Pass

Layer by layer: `z = W @ x + b`, then `a = activation(z)`. The output of one
layer is the input to the next.

### 3. Backward Pass (Backpropagation)

Starting from the loss, compute gradients layer by layer in reverse:

1. `dL/d(output)` -- from the loss function
2. `dL/d(W2)` = `dL/d(output) @ hidden.T`
3. `dL/d(hidden)` = `W2.T @ dL/d(output)` (then mask by ReLU derivative)
4. `dL/d(W1)` = `dL/d(hidden) @ x.T`

> **Math Callout:** This is the chain rule applied repeatedly. Each layer
> "passes back" the gradient it receives, scaled by its own local gradient.
> See 1D-02 for chain rule exercises.

### 4. Gradient Checking

Numerical approximation: `df/dx ~ (f(x+h) - f(x-h)) / (2h)`. Compare this
to your analytical gradient. If they differ by more than 1e-5 (relative), you
have a bug.

## Mini-Project: Classify Spirals

The spirals dataset has two interleaved spirals in 2D. A linear classifier
fails completely. Your two-layer network should separate them.

Target: >90% accuracy on the spiral dataset with a hidden layer of 100 units.

## "Wrong Way" Challenge

Try training with all weights initialized to zero. What happens? (Hint:
symmetry breaking -- all neurons compute the same thing.)

## Style Notes

- Implement forward and backward as separate functions, not methods on a class.
  This keeps the data flow explicit.
- Return intermediate values from the forward pass so the backward pass can
  use them (the "cache" pattern).

## Exercises

See [exercises.py](exercises.py) for stubs. Mini-project in
[mini_project.py](mini_project.py). Tests in `tests/`.
