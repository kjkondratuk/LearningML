# 05 -- CNNs for Images

## Learning Objectives

- Understand convolution as a sliding-window dot product
- Compute output sizes for conv and pooling layers
- Build a CNN in PyTorch from scratch
- Count parameters and understand the efficiency of weight sharing
- Train a CNN on CIFAR-10

## The Big Idea

A **convolutional neural network** exploits spatial structure. Instead of
connecting every input pixel to every neuron (fully connected), a CNN slides a
small filter across the image, computing dot products at each position. This
gives two massive advantages:

1. **Parameter sharing** -- the same filter is used everywhere, so a 3x3 filter
   has only 9 weights regardless of image size.
2. **Translation equivariance** -- a cat in the top-left activates the same
   filter as a cat in the bottom-right.

### Go Parallel

A convolution is a nested loop:

```go
for y := range outputH {
    for x := range outputW {
        output[y][x] = dotProduct(
            image[y:y+kH][x:x+kW],
            kernel,
        )
    }
}
```

PyTorch's `nn.Conv2d` does this (highly optimized via cuDNN) plus adds a bias
and stacks multiple filters.

## Key Concepts

### 1. Manual Conv2d

Implement convolution with explicit loops to understand what the operation does.
Then appreciate how much faster `nn.Conv2d` is.

### 2. Output Size Formula

```
output_size = (input_size - kernel_size + 2 * padding) / stride + 1
```

> **Math Callout:** This formula comes from counting how many positions the
> kernel fits. It is integer arithmetic -- if it does not divide evenly, you
> either pad or lose border pixels.

### 3. Typical CNN Architecture

```
Input -> [Conv -> ReLU -> Pool] x N -> Flatten -> FC -> Output
```

Each conv-pool block roughly halves spatial dimensions while increasing
channel depth.

### 4. Parameter Counting

For `nn.Conv2d(in_c, out_c, kernel_size=k)`:
```
params = out_c * (in_c * k * k + 1)   # +1 for bias
```

Compare to a fully connected layer on a 32x32x3 image: 3072 * hidden_size.
CNNs are dramatically more efficient.

## Style Notes

- Always check output shapes after each layer during development. Use
  `print(x.shape)` or a debug forward pass.
- Flatten with `x.view(x.size(0), -1)` before the fully connected layers.

## Exercises

See [exercises.py](exercises.py) for 5 stubs. Tests in
`tests/test_exercises.py`.
