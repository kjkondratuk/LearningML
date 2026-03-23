"""
Model Debugging Exercises -- Spiral 2, Phase 2D, Module 04
"""

import numpy as np
from typing import Callable


def overfit_single_batch(
    model_fn: Callable, X_batch: np.ndarray, y_batch: np.ndarray,
    lr: float = 0.01, n_steps: int = 200,
) -> list[float]:
    """The first debugging step: can the model memorize one batch?

    If loss does not approach 0, something is fundamentally broken.

    Returns:
        losses: loss at each step
    """
    raise NotImplementedError


def gradient_statistics(model_fn: Callable, X: np.ndarray, y: np.ndarray) -> dict:
    """Compute gradient statistics for each layer.

    Returns:
        dict mapping layer_name -> {mean, std, min, max, norm}
    """
    raise NotImplementedError


def activation_statistics(model_fn: Callable, X: np.ndarray) -> dict:
    """Compute activation statistics at each layer.

    Returns:
        dict mapping layer_name -> {mean, std, fraction_dead}
    """
    raise NotImplementedError


def learning_rate_sensitivity(
    model_fn: Callable, X: np.ndarray, y: np.ndarray,
    lr_values: list[float], n_epochs: int = 5,
) -> dict[float, list[float]]:
    """Train for a few epochs at each LR, record losses.

    Returns:
        dict mapping lr -> loss_history
    """
    raise NotImplementedError


def detect_data_leakage(
    X_train: np.ndarray, X_test: np.ndarray,
    feature_names: list[str] = None,
) -> dict:
    """Check for data leakage.

    Detect: suspiciously predictive features, duplicate rows across sets.

    Returns:
        dict with 'duplicate_rows', 'suspicious_features'
    """
    raise NotImplementedError


def training_diagnostics(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float] = None,
    val_accs: list[float] = None,
) -> str:
    """Analyze training curves and return diagnosis.

    Categories: "underfitting", "overfitting", "good_fit", "training_instability"

    Returns:
        diagnosis string with explanation
    """
    raise NotImplementedError


def ablation_study(
    model_fn: Callable,
    ablation_configs: dict[str, dict],
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, float]:
    """Train multiple model variants, each with one component removed.

    Args:
        model_fn: callable(**config) -> trained model
        ablation_configs: dict mapping variant_name -> config dict
        X, y: data

    Returns:
        dict mapping variant_name -> test score
    """
    raise NotImplementedError
