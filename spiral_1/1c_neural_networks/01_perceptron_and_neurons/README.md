# 01 -- Perceptron & Neurons

## Learning Objectives

- Understand the perceptron as a linear classifier with a threshold
- Implement the perceptron learning rule from scratch
- See *why* a single perceptron cannot solve XOR (and what that implies)
- Implement ReLU and softmax -- the two activations you will use most often

## The Big Idea

A **perceptron** is the simplest possible neural network: one neuron, one
decision boundary. It computes:

```
output = step(w . x + b)
```

where `w . x` is the dot product of weights and inputs, `b` is the bias, and
`step` returns 1 if the argument is >= 0, else 0.

### Go Parallel

Think of a perceptron as a function:

```go
func predict(weights, inputs []float64, bias float64) int {
    sum := bias
    for i := range weights {
        sum += weights[i] * inputs[i]
    }
    if sum >= 0 { return 1 }
    return 0
}
```

The "learning" part is a loop that adjusts `weights` and `bias` whenever the
prediction is wrong. That is literally the entire algorithm.

## Key Concepts

### 1. Step Function

The classic activation: returns 1 if input >= 0, else 0. Simple, but not
differentiable at 0, which is why modern networks use smoother alternatives.

### 2. Perceptron Learning Rule

For each misclassified sample:
```
w_new = w_old + lr * (target - predicted) * x
b_new = b_old + lr * (target - predicted)
```

This is guaranteed to converge **if the data is linearly separable**.

### 3. ReLU (Rectified Linear Unit)

```
relu(x) = max(0, x)
```

The workhorse of deep learning. Almost linear, easy to compute, and its
gradient is either 0 or 1 -- no vanishing gradient problem for positive inputs.

### 4. Softmax

Converts a vector of raw scores (logits) into a probability distribution:

```
softmax(z_i) = exp(z_i) / sum(exp(z_j))
```

> **Math Callout:** The softmax denominator is the normalizing constant that
> ensures the outputs sum to 1. This connects to the Boltzmann distribution
> in physics and the log-sum-exp trick in numerical computing (see 1D-03).

## "Wrong Way" Challenge: XOR with a Single Perceptron

Try to train a single perceptron on XOR:

| x1 | x2 | target |
|----|----|--------|
| 0  | 0  | 0      |
| 0  | 1  | 1      |
| 1  | 0  | 1      |
| 1  | 1  | 0      |

**It will never converge.** XOR is not linearly separable -- no single line
can divide the 1s from the 0s. This was the fundamental limitation that
Minsky & Papert identified in 1969, and it is the reason we need *multi-layer*
networks (covered in module 02).

## Style Notes

- Keep functions pure where possible. `step_function` takes a number, returns
  a number.
- Use NumPy arrays for weights and inputs, not Python lists.
- Type hints are encouraged: `def relu(x: np.ndarray) -> np.ndarray`.

## Exercises

See [exercises.py](exercises.py) for 5 function stubs. Tests are in
`tests/test_exercises.py`.
