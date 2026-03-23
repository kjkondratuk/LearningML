"""
Mini-Project: PCA + t-SNE on Fashion-MNIST
============================================

Spiral 2, Phase 2B, Module 07

1. Load Fashion-MNIST (10 classes)
2. PCA: explained variance plot, 2D visualization
3. t-SNE: 2D visualization, compare to PCA
4. PCA as preprocessing for t-SNE (PCA to 50 dims, then t-SNE to 2)
5. Reconstruct images from PCA at 10, 50, 100, 200 components
6. Compare wall-clock time: PCA vs t-SNE
"""

import numpy as np


def load_fashion_mnist(n_samples: int = 5000, seed: int = 42):
    """Load a subset of Fashion-MNIST.

    Returns:
        X: shape (n_samples, 784)
        y: shape (n_samples,), labels 0-9
    """
    raise NotImplementedError


def pca_2d_visualization(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Project data to 2D via PCA.

    Returns:
        X_2d: shape (n, 2)
    """
    raise NotImplementedError


def tsne_2d_visualization(
    X: np.ndarray, y: np.ndarray, perplexity: float = 30.0
) -> np.ndarray:
    """Project data to 2D via t-SNE.

    Returns:
        X_2d: shape (n, 2)
    """
    raise NotImplementedError


def pca_reconstruction_quality(
    X: np.ndarray, n_components_list: list[int]
) -> dict[int, float]:
    """Compute reconstruction error for different numbers of PCA components.

    Returns:
        dict mapping n_components -> relative reconstruction error
    """
    raise NotImplementedError
