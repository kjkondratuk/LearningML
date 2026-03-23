"""
Module 03: PyTorch Fundamentals
================================
Learn tensor operations, autograd, and nn.Module basics.
"""

import numpy as np
import torch
import torch.nn as nn


def numpy_to_torch_and_back(arr: np.ndarray) -> np.ndarray:
    """Convert a NumPy array to a PyTorch tensor, double every element, convert back.

    Steps:
        1. Convert arr to a torch tensor (float32).
        2. Multiply every element by 2.
        3. Convert back to a NumPy array and return.

    Args:
        arr: A NumPy array of any shape.

    Returns:
        A NumPy array with the same shape, all values doubled.
    """
    raise NotImplementedError


def tensor_operations(t: torch.Tensor) -> dict[str, torch.Tensor]:
    """Perform a suite of tensor operations and return results in a dict.

    Given a 2D tensor t of shape (M, N), return a dict with:
        - "transposed": t transposed (N, M)
        - "reshaped": t reshaped to (M * N,) (flattened)
        - "sum_cols": sum along columns, shape (M,)
        - "max_val": the single maximum value as a scalar tensor
        - "normalized": t with each row divided by its L2 norm

    Args:
        t: A 2D float tensor.

    Returns:
        Dict with the five keys described above.
    """
    raise NotImplementedError


def autograd_demo(x_val: float) -> dict[str, float]:
    """Demonstrate autograd by computing gradients of f(x) = x^3 + 2x^2 - 5x + 3.

    Steps:
        1. Create a tensor x with requires_grad=True from x_val.
        2. Compute y = x^3 + 2*x^2 - 5*x + 3.
        3. Call y.backward().
        4. Return {"y": y.item(), "dy_dx": x.grad.item()}.

    The analytical derivative is dy/dx = 3x^2 + 4x - 5.

    Args:
        x_val: The point at which to evaluate f and its derivative.

    Returns:
        Dict with keys "y" and "dy_dx".
    """
    raise NotImplementedError


class CustomLinearLayer(nn.Module):
    """A manually implemented linear layer: y = x @ W.T + b.

    This replicates what nn.Linear does under the hood.

    Args:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
    """

    def __init__(self, in_features: int, out_features: int):
        """Initialize weight and bias as nn.Parameter.

        W should have shape (out_features, in_features), initialized from
        a standard normal distribution scaled by 1/sqrt(in_features).
        b should have shape (out_features,), initialized to zeros.
        """
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute x @ W.T + b.

        Args:
            x: Tensor of shape (batch_size, in_features).

        Returns:
            Tensor of shape (batch_size, out_features).
        """
        raise NotImplementedError


def simple_model_forward(input_dim: int, hidden_dim: int, output_dim: int,
                         X: torch.Tensor) -> torch.Tensor:
    """Build a simple model (Linear -> ReLU -> Linear) and run a forward pass.

    Use nn.Sequential to stack:
        1. nn.Linear(input_dim, hidden_dim)
        2. nn.ReLU()
        3. nn.Linear(hidden_dim, output_dim)

    Do NOT train -- just initialize and run forward.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer size.
        output_dim: Output dimension.
        X: Input tensor of shape (batch_size, input_dim).

    Returns:
        Output tensor of shape (batch_size, output_dim).
    """
    raise NotImplementedError
