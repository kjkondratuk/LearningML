"""
Tests for Backpropagation exercises.
"""

import numpy as np
import pytest

from exercises import (
    forward_pass,
    backward_pass,
    numerical_gradient_check,
    mse_loss_and_grad,
    cross_entropy_loss_and_grad,
    softmax_cross_entropy_backward,
    train_neural_network,
)


def relu(z):
    return np.maximum(0, z)

def relu_grad(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)

def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class TestGradientCheck:
    """Gradient check should pass for all layers (relative error < 1e-5)."""

    def test_2_layer_network(self):
        np.random.seed(42)
        X = np.random.randn(5, 3)
        y = np.random.randn(5, 2)
        W1 = np.random.randn(3, 4) * 0.1
        b1 = np.zeros(4)
        W2 = np.random.randn(4, 2) * 0.1
        b2 = np.zeros(2)

        weights = [W1, W2]
        biases = [b1, b2]

        out, cached_z, cached_a = forward_pass(
            X, weights, biases, [relu, lambda z: z]
        )
        loss, loss_grad = mse_loss_and_grad(out, y)

        w_grads, b_grads = backward_pass(
            y, weights, cached_z, cached_a,
            [relu_grad, lambda z: np.ones_like(z)],
            lambda y_pred, y_true: (2.0 / y_pred.shape[0]) * (y_pred - y_true),
        )

        # Numerical check for W1
        def f_w1(W1_flat):
            W1_r = W1_flat.reshape(W1.shape)
            out, _, _ = forward_pass(X, [W1_r, W2], [b1, b2], [relu, lambda z: z])
            loss, _ = mse_loss_and_grad(out, y)
            return loss

        num_grads = numerical_gradient_check(
            lambda params: f_w1(params[0].flatten()),
            [W1],
        )
        rel_error = np.abs(num_grads[0] - w_grads[0]) / (np.abs(num_grads[0]) + np.abs(w_grads[0]) + 1e-8)
        assert np.max(rel_error) < 1e-4


class TestSoftmaxCrossEntropy:
    def test_gradient_matches_numerical(self):
        np.random.seed(42)
        logits = np.random.randn(4, 3)
        y_true = np.zeros((4, 3))
        y_true[np.arange(4), [0, 1, 2, 1]] = 1.0

        grad = softmax_cross_entropy_backward(logits, y_true)

        # Numerical check
        eps = 1e-5
        num_grad = np.zeros_like(logits)
        for i in range(logits.shape[0]):
            for j in range(logits.shape[1]):
                logits_p = logits.copy()
                logits_p[i, j] += eps
                logits_m = logits.copy()
                logits_m[i, j] -= eps
                s_p = softmax(logits_p)
                s_m = softmax(logits_m)
                loss_p = -np.sum(y_true * np.log(s_p + 1e-10)) / logits.shape[0]
                loss_m = -np.sum(y_true * np.log(s_m + 1e-10)) / logits.shape[0]
                num_grad[i, j] = (loss_p - loss_m) / (2 * eps)

        np.testing.assert_allclose(grad, num_grad, atol=1e-4)


class TestTrainNetwork:
    def test_learns_xor(self):
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([[0], [1], [1], [0]], dtype=float)
        weights, biases, losses = train_neural_network(
            X, y, layer_sizes=[2, 8, 1], lr=0.5, n_epochs=2000, batch_size=4
        )
        assert losses[-1] < 0.05

    def test_loss_decreases(self):
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (X[:, 0] > 0).astype(float).reshape(-1, 1)
        _, _, losses = train_neural_network(
            X, y, layer_sizes=[5, 10, 1], lr=0.01, n_epochs=100, batch_size=32
        )
        assert losses[-1] < losses[0]
