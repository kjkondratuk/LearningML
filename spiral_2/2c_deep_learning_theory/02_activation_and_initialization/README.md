# Module 02: Activation Functions and Weight Initialization

> The vanishing gradient problem is the reason deep networks were impractical for
> decades. This module derives the problem and its solutions: ReLU fixes the
> gradient, Xavier/He initialization fixes the scale.

## Learning Objectives

- Implement sigmoid, tanh, ReLU, leaky ReLU, GELU and their gradients.
- Demonstrate the vanishing gradient problem with deep sigmoid networks.
- Derive Xavier initialization from the variance-preservation condition.
- Derive He initialization for ReLU networks.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Goodfellow, *Deep Learning*, ch 6.2--6.3 | Hidden units, design |
| Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010) | Xavier init derivation |
| He et al., "Delving Deep into Rectifiers" (2015) | He init for ReLU |

## Derive It

1. **Sigmoid gradient.** Show sigma'(z) = sigma(z)(1 - sigma(z)). Max = 0.25 at z=0.
   For a chain of L layers, gradient scales as (0.25)^L. At L=20, this is ~1e-12.

2. **Xavier initialization.** For linear activation y = Wx, require Var(y) = Var(x).
   Show this requires Var(w) = 2/(fan_in + fan_out).

3. **He initialization.** For ReLU activation, half the outputs are zero. Show this
   requires Var(w) = 2/fan_in.

## "Naive then Derive" Challenge

Initialize a 20-layer sigmoid network with N(0,1) weights. Forward-pass and observe
activations collapsing to 0. Apply Xavier init and see them stay healthy. Repeat
with ReLU + He init.

## Exercises

See `exercises.py`.
