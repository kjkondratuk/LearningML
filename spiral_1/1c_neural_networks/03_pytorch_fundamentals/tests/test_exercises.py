"""
Tests for Module 03: PyTorch Fundamentals

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import torch
import pytest

from exercises import (
    numpy_to_torch_and_back,
    tensor_operations,
    autograd_demo,
    CustomLinearLayer,
    simple_model_forward,
)


# ---------------------------------------------------------------------------
# numpy_to_torch_and_back
# ---------------------------------------------------------------------------

class TestNumpyToTorchAndBack:
    def test_values_doubled(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = numpy_to_torch_and_back(arr)
        np.testing.assert_allclose(result, np.array([2.0, 4.0, 6.0]))

    def test_returns_numpy(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = numpy_to_torch_and_back(arr)
        assert isinstance(result, np.ndarray)

    def test_preserves_shape(self):
        arr = np.random.randn(3, 4, 5)
        result = numpy_to_torch_and_back(arr)
        assert result.shape == (3, 4, 5)

    def test_zeros(self):
        arr = np.zeros((2, 2))
        result = numpy_to_torch_and_back(arr)
        np.testing.assert_array_equal(result, np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# tensor_operations
# ---------------------------------------------------------------------------

class TestTensorOperations:
    def setup_method(self):
        self.t = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    def test_transposed_shape(self):
        result = tensor_operations(self.t)
        assert result["transposed"].shape == (3, 2)

    def test_reshaped_is_flat(self):
        result = tensor_operations(self.t)
        assert result["reshaped"].shape == (6,)

    def test_sum_cols(self):
        result = tensor_operations(self.t)
        expected = torch.tensor([6.0, 15.0])
        torch.testing.assert_close(result["sum_cols"], expected)

    def test_max_val(self):
        result = tensor_operations(self.t)
        assert result["max_val"].item() == 6.0

    def test_normalized_rows_unit_norm(self):
        result = tensor_operations(self.t)
        norms = result["normalized"].norm(dim=1)
        torch.testing.assert_close(norms, torch.ones(2), atol=1e-6, rtol=0)


# ---------------------------------------------------------------------------
# autograd_demo
# ---------------------------------------------------------------------------

class TestAutogradDemo:
    def test_at_zero(self):
        result = autograd_demo(0.0)
        # f(0) = 0 + 0 - 0 + 3 = 3
        assert abs(result["y"] - 3.0) < 1e-5
        # f'(0) = 0 + 0 - 5 = -5
        assert abs(result["dy_dx"] - (-5.0)) < 1e-5

    def test_at_one(self):
        result = autograd_demo(1.0)
        # f(1) = 1 + 2 - 5 + 3 = 1
        assert abs(result["y"] - 1.0) < 1e-5
        # f'(1) = 3 + 4 - 5 = 2
        assert abs(result["dy_dx"] - 2.0) < 1e-5

    def test_at_negative(self):
        result = autograd_demo(-2.0)
        # f(-2) = -8 + 8 + 10 + 3 = 13
        assert abs(result["y"] - 13.0) < 1e-5
        # f'(-2) = 12 - 8 - 5 = -1
        assert abs(result["dy_dx"] - (-1.0)) < 1e-5

    def test_returns_dict(self):
        result = autograd_demo(0.0)
        assert isinstance(result, dict)
        assert "y" in result
        assert "dy_dx" in result


# ---------------------------------------------------------------------------
# CustomLinearLayer
# ---------------------------------------------------------------------------

class TestCustomLinearLayer:
    def test_output_shape(self):
        layer = CustomLinearLayer(10, 5)
        x = torch.randn(3, 10)
        out = layer(x)
        assert out.shape == (3, 5)

    def test_has_parameters(self):
        layer = CustomLinearLayer(10, 5)
        param_names = [name for name, _ in layer.named_parameters()]
        assert any('W' in name or 'weight' in name for name in param_names)

    def test_gradients_flow(self):
        layer = CustomLinearLayer(4, 2)
        x = torch.randn(1, 4, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None

    def test_bias_initialized_zero(self):
        layer = CustomLinearLayer(10, 5)
        # Find the bias parameter
        for name, param in layer.named_parameters():
            if 'b' in name.lower() or 'bias' in name.lower():
                torch.testing.assert_close(
                    param.data, torch.zeros_like(param.data)
                )
                break


# ---------------------------------------------------------------------------
# simple_model_forward
# ---------------------------------------------------------------------------

class TestSimpleModelForward:
    def test_output_shape(self):
        X = torch.randn(8, 10)
        out = simple_model_forward(10, 20, 3, X)
        assert out.shape == (8, 3)

    def test_returns_tensor(self):
        X = torch.randn(4, 5)
        out = simple_model_forward(5, 10, 2, X)
        assert isinstance(out, torch.Tensor)

    def test_different_dims(self):
        X = torch.randn(1, 100)
        out = simple_model_forward(100, 50, 10, X)
        assert out.shape == (1, 10)
