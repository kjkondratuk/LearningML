"""
Data Pipelines Exercises -- Spiral 2, Phase 2D, Module 03
"""

import numpy as np
from typing import Callable, Optional


def custom_dataset(data_path: str, transform_fn: Optional[Callable] = None):
    """Implement a PyTorch-style Dataset with __len__ and __getitem__.

    Include lazy loading (load from disk on demand, not all at once).

    Args:
        data_path: path to data
        transform_fn: optional transform applied to each item

    Returns:
        dataset object with __len__ and __getitem__
    """
    raise NotImplementedError


def data_augmentation_pipeline(image: np.ndarray, seed: int = None) -> np.ndarray:
    """Implement common augmentations from scratch.

    Apply (randomly): random crop, horizontal flip, color jitter, normalization.

    Args:
        image: shape (H, W, C), uint8 or float

    Returns:
        augmented: shape (H, W, C)
    """
    raise NotImplementedError


def stratified_split(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple:
    """Split data maintaining class proportions. No sklearn.

    Args:
        X, y: data and labels
        test_ratio: fraction for test
        val_ratio: fraction for validation
        seed: random seed

    Returns:
        (X_train, y_train, X_val, y_val, X_test, y_test)
    """
    raise NotImplementedError


def efficient_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """Wrap a dataset in a PyTorch DataLoader.

    Benchmark different num_workers values.

    Args:
        dataset: PyTorch Dataset
        batch_size: batch size
        shuffle: whether to shuffle
        num_workers: parallel data loading workers

    Returns:
        DataLoader instance
    """
    raise NotImplementedError


def handle_imbalanced_data(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "oversample",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Handle class imbalance.

    Strategies:
    - "oversample": duplicate minority class samples
    - "undersample": remove majority class samples
    - "smote": simplified SMOTE (interpolate between minority neighbors)

    Args:
        X, y: imbalanced data
        strategy: "oversample", "undersample", or "smote"
        seed: random seed

    Returns:
        X_balanced, y_balanced
    """
    raise NotImplementedError


def data_validation(X: np.ndarray, y: np.ndarray) -> dict:
    """Validate data quality.

    Check for: NaN values, constant features, duplicate rows,
    class imbalance, feature scale issues.

    Args:
        X, y: data to validate

    Returns:
        report: dict with 'nan_count', 'constant_features', 'duplicate_rows',
                'class_distribution', 'scale_issues'
    """
    raise NotImplementedError
