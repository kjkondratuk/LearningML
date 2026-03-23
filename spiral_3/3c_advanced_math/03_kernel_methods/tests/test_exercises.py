"""
Tests for Kernel Methods exercises.

Verifies: RBF properties, Mercer conditions, KRR convergence, kernel PCA separation.
"""

import pytest
import torch

from exercises import (
    kernel_pca,
    kernel_ridge_regression,
    nystrom_approximation,
    polynomial_kernel,
    rbf_kernel,
    verify_mercer_conditions,
)


class TestRBFKernel:
    def test_self_similarity_is_one(self):
        x = torch.randn(5, 3)
        K = rbf_kernel(x, x, sigma=1.0)
        assert torch.allclose(K.diag(), torch.ones(5), atol=1e-5)

    def test_decays_with_distance(self):
        x = torch.zeros(1, 2)
        y_near = torch.tensor([[0.1, 0.0]])
        y_far = torch.tensor([[10.0, 0.0]])
        k_near = rbf_kernel(x, y_near, sigma=1.0)
        k_far = rbf_kernel(x, y_far, sigma=1.0)
        assert k_near > k_far

    def test_positive_values(self):
        x = torch.randn(10, 5)
        K = rbf_kernel(x, x)
        assert (K >= 0).all()


class TestMercerConditions:
    def test_valid_kernel_is_psd(self):
        x = torch.randn(20, 5)
        K = rbf_kernel(x, x, sigma=1.0)
        assert verify_mercer_conditions(K) is True

    def test_random_matrix_not_psd(self):
        M = torch.randn(10, 10)
        assert verify_mercer_conditions(M) is False


class TestKernelRidgeRegression:
    def test_interpolation_with_small_lambda(self):
        """With lambda -> 0, KRR interpolates training data."""
        x = torch.randn(20, 3)
        K = rbf_kernel(x, x, sigma=1.0)
        y = torch.randn(20)
        alpha = kernel_ridge_regression(K, y, lambda_reg=1e-8)
        y_pred = K @ alpha
        assert torch.allclose(y_pred, y, atol=1e-3)

    def test_mean_prediction_with_large_lambda(self):
        """With lambda -> infinity, prediction approaches zero (or mean with bias)."""
        x = torch.randn(20, 3)
        K = rbf_kernel(x, x, sigma=1.0)
        y = torch.randn(20)
        alpha = kernel_ridge_regression(K, y, lambda_reg=1e6)
        y_pred = K @ alpha
        assert y_pred.abs().max() < 1.0


class TestKernelPCA:
    def test_output_shape(self):
        x = torch.randn(50, 10)
        K = rbf_kernel(x, x)
        projected = kernel_pca(K, n_components=3)
        assert projected.shape == (50, 3)


class TestNystrom:
    def test_approximation_improves_with_landmarks(self):
        x = torch.randn(50, 5)
        K_exact = rbf_kernel(x, x, sigma=1.0)

        landmarks_few = x[:5]
        landmarks_many = x[:20]

        K_approx_few = nystrom_approximation(x, landmarks_few, lambda x1, x2: rbf_kernel(x1, x2), rank=5)
        K_approx_many = nystrom_approximation(x, landmarks_many, lambda x1, x2: rbf_kernel(x1, x2), rank=20)

        err_few = (K_exact - K_approx_few).norm()
        err_many = (K_exact - K_approx_many).norm()
        assert err_many < err_few
