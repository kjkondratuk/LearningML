"""
Tests for the Computational Graph Engine mini-project.
"""

import numpy as np
import pytest

from mini_project import Tensor


class TestTensorOps:
    def test_add(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = a + b
        np.testing.assert_array_equal(c.data, [4.0, 6.0])

    def test_mul(self):
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, 5.0])
        c = a * b
        np.testing.assert_array_equal(c.data, [8.0, 15.0])

    def test_relu(self):
        a = Tensor([-1.0, 0.0, 2.0])
        b = a.relu()
        np.testing.assert_array_equal(b.data, [0.0, 0.0, 2.0])


class TestBackward:
    def test_simple_gradient(self):
        """f = (a * b).sum(), df/da = b."""
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, 5.0])
        c = a * b
        d = c.sum()
        d.backward()
        np.testing.assert_array_equal(a.grad, [4.0, 5.0])
        np.testing.assert_array_equal(b.grad, [2.0, 3.0])

    def test_matmul_gradient(self):
        """f = sum(x @ W), check gradients."""
        x = Tensor(np.array([[1.0, 2.0]]))
        W = Tensor(np.array([[3.0], [4.0]]))
        y = x.matmul(W)
        z = y.sum()
        z.backward()
        # dz/dW = x^T
        np.testing.assert_allclose(W.grad, [[1.0], [2.0]], atol=1e-10)

    def test_relu_gradient(self):
        a = Tensor([-1.0, 2.0, -3.0, 4.0])
        b = a.relu()
        c = b.sum()
        c.backward()
        np.testing.assert_array_equal(a.grad, [0.0, 1.0, 0.0, 1.0])
