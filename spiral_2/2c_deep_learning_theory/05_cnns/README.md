# Module 05: Convolutional Neural Networks

> Convolution is the right inductive bias for images: translation equivariance
> and parameter sharing. This module implements conv from scratch in NumPy,
> including the im2col trick that makes it fast.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Goodfellow, *Deep Learning*, ch 9 | Convolutional networks |
| CS231n lecture notes on ConvNets (Stanford) | Practical treatment |
| Dumoulin & Visin, "A guide to convolution arithmetic" (2016) | Excellent diagrams |

## Derive It

1. **Convolution as correlation.** Show that conv(f, g) slides g over f and computes
   dot products. In CNNs we actually compute cross-correlation (no flip).

2. **im2col trick.** Reshape input patches into columns. Now convolution = matrix
   multiply on the reshaped input. Derive the reshaping.

3. **Backward pass.** Show that the gradient w.r.t. the kernel is also a convolution.
   The gradient w.r.t. the input is a "full" convolution with the flipped kernel.

4. **Receptive field.** For a stack of 3x3 convolutions, derive the receptive field
   at each layer. Three 3x3 layers have the same receptive field as one 7x7 layer.

## "Naive then Derive" Challenge

Implement conv2d with 4 nested for-loops. Time it. Then implement im2col and compare.

## Exercises

See `exercises.py`.

## Mini-Project: CNN from Scratch in NumPy

Build a 3-layer CNN, train on MNIST, visualize filters and feature maps.
See `mini_project.py`.
