"""
Tests for Module 02: Neural Net from Scratch

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    initialize_weights,
    forward_pass,
    backward_pass,
    update_params,
    train_network,
)


# ---------------------------------------------------------------------------
# initialize_weights
# ---------------------------------------------------------------------------

class TestInitializeWeights:
    def test_correct_number_of_layers(self):
        params = initialize_weights([2, 100, 3])
        assert len(params) == 2

    def test_weight_shapes(self):
        params = initialize_weights([4, 10, 5, 3])
        assert params[0]['W'].shape == (10, 4)
        assert params[0]['b'].shape == (10, 1)
        assert params[1]['W'].shape == (5, 10)
        assert params[1]['b'].shape == (5, 1)
        assert params[2]['W'].shape == (3, 5)
        assert params[2]['b'].shape == (3, 1)

    def test_biases_are_zero(self):
        params = initialize_weights([2, 10, 3])
        for p in params:
            np.testing.assert_array_equal(p['b'], np.zeros_like(p['b']))

    def test_he_initialization_scale(self):
        """Weights should have std approximately sqrt(2/fan_in)."""
        params = initialize_weights([1000, 500], seed=0)
        W = params[0]['W']
        expected_std = np.sqrt(2.0 / 1000)
        assert abs(W.std() - expected_std) < 0.05

    def test_reproducible(self):
        p1 = initialize_weights([2, 10], seed=42)
        p2 = initialize_weights([2, 10], seed=42)
        np.testing.assert_array_equal(p1[0]['W'], p2[0]['W'])


# ---------------------------------------------------------------------------
# forward_pass
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_output_shape(self):
        params = initialize_weights([2, 10, 3])
        X = np.random.randn(2, 5)
        out, cache = forward_pass(X, params)
        assert out.shape == (3, 5)

    def test_output_sums_to_one(self):
        params = initialize_weights([2, 10, 3])
        X = np.random.randn(2, 5)
        out, _ = forward_pass(X, params)
        np.testing.assert_allclose(out.sum(axis=0), np.ones(5), atol=1e-6)

    def test_cache_length(self):
        params = initialize_weights([2, 10, 3])
        X = np.random.randn(2, 5)
        _, cache = forward_pass(X, params)
        # Cache should have entries for input + each layer
        assert len(cache) >= 2


# ---------------------------------------------------------------------------
# backward_pass -- numerical gradient check
# ---------------------------------------------------------------------------

class TestBackwardPass:
    def _numerical_grad(self, params, X, Y, layer_idx, key, h=1e-5):
        """Compute numerical gradient for one parameter matrix."""
        param = params[layer_idx][key]
        grad = np.zeros_like(param)
        for i in range(param.shape[0]):
            for j in range(param.shape[1]):
                old_val = param[i, j]
                param[i, j] = old_val + h
                out_plus, _ = forward_pass(X, params)
                loss_plus = -np.sum(Y * np.log(out_plus + 1e-8)) / X.shape[1]
                param[i, j] = old_val - h
                out_minus, _ = forward_pass(X, params)
                loss_minus = -np.sum(Y * np.log(out_minus + 1e-8)) / X.shape[1]
                grad[i, j] = (loss_plus - loss_minus) / (2 * h)
                param[i, j] = old_val
        return grad

    def test_gradient_check_W(self):
        np.random.seed(0)
        params = initialize_weights([2, 5, 3], seed=0)
        X = np.random.randn(2, 4)
        Y = np.eye(3)[:, np.array([0, 1, 2, 0])]
        out, cache = forward_pass(X, params)
        grads = backward_pass(Y, out, params, cache)

        for l in range(len(params)):
            num_grad = self._numerical_grad(params, X, Y, l, 'W')
            np.testing.assert_allclose(
                grads[l]['dW'], num_grad, atol=1e-5,
                err_msg=f"Gradient check failed for layer {l} W"
            )

    def test_gradient_check_b(self):
        np.random.seed(0)
        params = initialize_weights([2, 5, 3], seed=0)
        X = np.random.randn(2, 4)
        Y = np.eye(3)[:, np.array([0, 1, 2, 0])]
        out, cache = forward_pass(X, params)
        grads = backward_pass(Y, out, params, cache)

        for l in range(len(params)):
            num_grad = self._numerical_grad(params, X, Y, l, 'b')
            np.testing.assert_allclose(
                grads[l]['db'], num_grad, atol=1e-5,
                err_msg=f"Gradient check failed for layer {l} b"
            )


# ---------------------------------------------------------------------------
# update_params
# ---------------------------------------------------------------------------

class TestUpdateParams:
    def test_weights_change(self):
        params = initialize_weights([2, 10, 3])
        grads = [
            {'dW': np.ones_like(p['W']), 'db': np.ones_like(p['b'])}
            for p in params
        ]
        new_params = update_params(params, grads, lr=0.1)
        for old, new in zip(params, new_params):
            assert not np.array_equal(old['W'], new['W'])

    def test_does_not_modify_original(self):
        params = initialize_weights([2, 10, 3])
        original_W = params[0]['W'].copy()
        grads = [
            {'dW': np.ones_like(p['W']), 'db': np.ones_like(p['b'])}
            for p in params
        ]
        update_params(params, grads, lr=0.1)
        np.testing.assert_array_equal(params[0]['W'], original_W)


# ---------------------------------------------------------------------------
# train_network
# ---------------------------------------------------------------------------

class TestTrainNetwork:
    def test_loss_decreases(self):
        np.random.seed(0)
        X = np.random.randn(2, 20)
        Y = np.eye(2)[:, (X[0] > 0).astype(int)]
        _, loss_history = train_network(X, Y, [2, 10, 2], lr=0.1, epochs=200)
        assert loss_history[-1] < loss_history[0]

    def test_returns_correct_types(self):
        X = np.random.randn(2, 10)
        Y = np.eye(2)[:, np.random.randint(0, 2, 10)]
        params, loss_history = train_network(X, Y, [2, 5, 2], epochs=10)
        assert isinstance(params, list)
        assert isinstance(loss_history, list)
        assert len(loss_history) == 10
