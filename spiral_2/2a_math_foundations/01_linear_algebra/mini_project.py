"""
Mini-Project: Image Compression via SVD
========================================

Spiral 2, Phase 2A, Module 01

Load a grayscale image, compute its SVD, and reconstruct using only the
top-k singular values for various values of k. Explore:

1. Reconstruction quality (Frobenius norm error) vs k
2. Compressed size vs k
3. Visual quality at different k values
4. Finding the "elbow" -- minimum k for visually acceptable quality

Use YOUR svd() implementation from exercises.py (or fall back to np.linalg.svd
for large images if your implementation is too slow).
"""

import numpy as np


def load_grayscale_image(path: str) -> np.ndarray:
    """Load an image and convert to a grayscale float matrix in [0, 1].

    Args:
        path: path to image file

    Returns:
        image: shape (H, W), float64 values in [0, 1]
    """
    raise NotImplementedError


def svd_compress(image: np.ndarray, k: int) -> np.ndarray:
    """Reconstruct image using only the top-k singular values.

    Given SVD: image = U @ diag(S) @ Vt, the rank-k approximation is:
        image_k = U[:, :k] @ diag(S[:k]) @ Vt[:k, :]

    Args:
        image: shape (H, W) grayscale image
        k: number of singular values to keep

    Returns:
        reconstructed: shape (H, W), the rank-k approximation
    """
    raise NotImplementedError


def compression_ratio(image: np.ndarray, k: int) -> float:
    """Compute the compression ratio for rank-k approximation.

    Original storage: H * W values
    Compressed storage: k * (H + W + 1) values (U[:,:k], S[:k], Vt[:k,:])

    Args:
        image: shape (H, W)
        k: number of singular values kept

    Returns:
        ratio: original_size / compressed_size
    """
    raise NotImplementedError


def reconstruction_error(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute the relative Frobenius norm error.

    error = ||original - reconstructed||_F / ||original||_F

    Args:
        original: shape (H, W)
        reconstructed: shape (H, W)

    Returns:
        relative error as a float
    """
    raise NotImplementedError


def find_elbow(image: np.ndarray, threshold: float = 0.05) -> int:
    """Find minimum k such that reconstruction error is below threshold.

    Args:
        image: shape (H, W) grayscale image
        threshold: maximum acceptable relative Frobenius norm error

    Returns:
        k: minimum number of singular values needed
    """
    raise NotImplementedError
