"""Tests for LASSO mini-project."""

import pytest
import torch

from mini_project import lasso_ista, lasso_fista


class TestLASSO:
    def test_ista_produces_sparse_solution(self):
        torch.manual_seed(42)
        X = torch.randn(50, 10)
        y = X[:, :2] @ torch.tensor([1.0, -1.0]) + 0.01 * torch.randn(50)
        w = lasso_ista(X, y, lambda_reg=0.1, iterations=2000)
        assert (w.abs() < 1e-3).sum() >= 5, "LASSO should zero out irrelevant features"
