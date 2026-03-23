# Module 01: Backpropagation Derived and Implemented

> Backpropagation is the chain rule applied to computational graphs. This module
> derives it from scratch, implements it for arbitrary architectures, and builds
> a simple autograd engine.

## Learning Objectives

- Derive the chain rule for a multi-layer network on paper.
- Implement forward and backward passes for arbitrary depth.
- Implement numerical gradient checking (the gold standard for debugging).
- Understand the combined softmax + cross-entropy gradient simplification.
- Build a simple computational graph autograd engine.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Goodfellow, *Deep Learning*, ch 6.5 | Back-propagation algorithm |
| 3Blue1Brown, "Backpropagation" and "Backpropagation calculus" (NN ch 3--4) | Visual chain rule |
| Colah, "Calculus on Computational Graphs: Backpropagation" (blog) | Graph perspective |
| Karpathy, "micrograd" video lecture | Building an autograd engine |

## Derive It

1. **Single layer.** For y = sigma(Wx + b), L = loss(y, target), derive:
   dL/dW = dL/dy * dy/dz * dz/dW where z = Wx + b.

2. **Two layers.** Chain through: dL/dW1 involves dL/dy2 * dy2/dz2 * dz2/dh1 * dh1/dz1 * dz1/dW1.
   The recursive structure suggests the algorithm.

3. **General backprop.** For layer l: delta_l = (W_{l+1}^T delta_{l+1}) * sigma'(z_l).
   dL/dW_l = delta_l @ a_{l-1}^T. This is the backpropagation algorithm.

4. **Softmax + cross-entropy.** Show that the gradient simplifies to
   (softmax(z) - y_one_hot). This cancellation makes implementation cleaner
   and more numerically stable.

## "Naive then Derive" Challenge

Manually write out all partial derivatives for a specific 2-layer network.
Then derive the general algorithm. Show they produce the same gradients.

## Exercises

See `exercises.py`.

## Mini-Project: Computational Graph Engine

Build a micrograd-style autograd engine extended to matrices. See `mini_project.py`.
