"""
Mini-Project: Regularization Ablation Study
=============================================

Spiral 2, Phase 2C, Module 04

Train 6 network variants on a small dataset (easy to overfit):
1. No regularization
2. Dropout only
3. Batch norm only
4. Weight decay only
5. Dropout + batch norm
6. All three

Compare: training/validation loss curves, final test accuracy.
"""

import numpy as np


def create_overfit_dataset(
    n_train: int = 100,
    n_test: int = 500,
    d: int = 50,
    seed: int = 42,
) -> tuple:
    """Create a dataset that is easy to overfit (small train, large model)."""
    raise NotImplementedError


def train_variant(
    X_train, y_train, X_val, y_val,
    use_dropout: bool = False,
    use_batch_norm: bool = False,
    use_weight_decay: bool = False,
    n_epochs: int = 200,
    seed: int = 42,
) -> dict:
    """Train one regularization variant.

    Returns:
        dict with 'train_losses', 'val_losses', 'test_accuracy'
    """
    raise NotImplementedError


def run_ablation() -> dict[str, dict]:
    """Run all 6 variants and collect results.

    Returns:
        dict mapping variant_name -> result dict
    """
    raise NotImplementedError
