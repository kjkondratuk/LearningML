"""
Mini-Project: Computational Graph Engine
==========================================

Spiral 2, Phase 2C, Module 01

Build a simple autograd engine (inspired by Karpathy's micrograd, extended to matrices):
1. Tensor class with forward operations: add, mul, matmul, relu, sigmoid, log
2. Each operation records itself in a computation graph
3. backward() traverses the graph and computes all gradients
4. Demo: define a neural network, train it, verify gradients match manual backprop
"""

import numpy as np


class Tensor:
    """A tensor that tracks its computation graph for automatic differentiation.

    Attributes:
        data: numpy array of values
        grad: numpy array of gradients (same shape as data)
        _backward: function to compute gradients of parents
        _parents: set of parent Tensors in the computation graph
    """

    def __init__(self, data, _parents=(), _op=''):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._parents = set(_parents)
        self._op = _op

    def __add__(self, other):
        """Element-wise addition with gradient tracking."""
        raise NotImplementedError

    def __mul__(self, other):
        """Element-wise multiplication with gradient tracking."""
        raise NotImplementedError

    def matmul(self, other):
        """Matrix multiplication with gradient tracking."""
        raise NotImplementedError

    def relu(self):
        """ReLU activation with gradient tracking."""
        raise NotImplementedError

    def sigmoid(self):
        """Sigmoid activation with gradient tracking."""
        raise NotImplementedError

    def log(self):
        """Element-wise log with gradient tracking."""
        raise NotImplementedError

    def sum(self):
        """Sum all elements, returning a scalar Tensor."""
        raise NotImplementedError

    def backward(self):
        """Compute gradients via reverse-mode autodiff (backpropagation).

        Topologically sort the computation graph, then traverse in reverse,
        calling _backward() at each node.
        """
        raise NotImplementedError

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"
