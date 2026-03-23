# Phase 1C: Neural Networks (Intuition Pass)

## Goal

Build neural networks from first principles, then graduate to PyTorch. By the end
of this phase you will have trained an image classifier that beats 85% accuracy on
CIFAR-10 using transfer learning -- and you will understand *why* every piece works.

## Prerequisites

| You should be comfortable with ...        | Covered in |
|-------------------------------------------|------------|
| NumPy array operations                    | 1B-01      |
| Gradient descent intuition                | 1D-04      |
| Matrix multiplication                     | 1D-01      |
| Basic probability (softmax denominator)   | 1D-03      |

## Module Index

| # | Module | Key Ideas | Exercises |
|---|--------|-----------|-----------|
| 01 | [Perceptron & Neurons](01_perceptron_and_neurons/) | Step function, perceptron learning rule, activation functions | 5 stubs + XOR "wrong way" |
| 02 | [Neural Net from Scratch](02_neural_net_from_scratch/) | Forward/backward pass, weight init, gradient checking | 5 stubs + spiral classifier mini-project |
| 03 | [PyTorch Fundamentals](03_pytorch_fundamentals/) | Tensors, autograd, nn.Module | 5 stubs |
| 04 | [Training Loop](04_training_loop/) | DataLoader, optimizer, train/eval split | 6 stubs + MNIST mini-project |
| 05 | [CNNs for Images](05_cnns_for_images/) | Convolutions, pooling, feature maps | 5 stubs |
| 06 | [RNNs for Sequences](06_rnns_for_sequences/) | Recurrence, LSTM gates, text classification | 5 stubs |
| 07 | [Transfer Learning](07_transfer_learning/) | Pretrained models, freezing, fine-tuning | 5 stubs |
| 08 | [Capstone: Image Classifier](08_capstone_image_classifier/) | End-to-end pipeline, model comparison | Full project |

## How to Work Through This Phase

1. **Do modules 01-02 with pen and paper nearby.** Drawing the forward pass
   helps enormously.
2. **Module 03 is a pivot point.** Everything before it is pure NumPy;
   everything after uses PyTorch.
3. **Modules 05-07 are "breadth" modules.** Get the intuition; deep mastery
   comes in Spiral 2.
4. **The capstone (08) ties it all together.** You will build two models and
   compare them quantitatively.

## Go Parallel

If you are coming from Go, think of a neural network layer as a
`func(input []float64) []float64` -- a pure function that transforms a slice.
Training is just adjusting the closure variables (weights) so the output
matches the target. PyTorch's autograd is like having a compiler that
automatically writes the derivative of every function you compose.

## Running Tests

```bash
# Run all 1C tests
python -m pytest spiral_1/1c_neural_networks/ -v

# Run a single module
python -m pytest spiral_1/1c_neural_networks/01_perceptron_and_neurons/tests/ -v
```
