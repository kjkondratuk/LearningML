"""
Module 08: Feature Engineering — Exercise Stubs

Implement each function until `pytest tests/test_exercises.py` passes.
"""

import numpy as np


def encode_ordinal(
    values: np.ndarray, mapping: dict[str, int]
) -> np.ndarray:
    """Map categorical string values to ordered integers.

    Parameters
    ----------
    values : np.ndarray of shape (n_samples,), string values
    mapping : dict mapping each category string to an integer

    Returns
    -------
    np.ndarray of shape (n_samples,), integer-encoded values
    """
    raise NotImplementedError


def encode_onehot_manual(
    values: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """One-hot encode a 1D array of categorical values.

    Parameters
    ----------
    values : np.ndarray of shape (n_samples,), string or int categories

    Returns
    -------
    encoded : np.ndarray of shape (n_samples, n_categories), binary matrix
    categories : list[str], the category names in column order (sorted)
    """
    raise NotImplementedError


def scale_features(
    X: np.ndarray,
    mean: np.ndarray = None,
    std: np.ndarray = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features to zero mean and unit variance.

    If mean and std are provided, use them (for transforming test data).
    Otherwise, compute from X (for fitting on training data).

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    mean : np.ndarray of shape (n_features,) or None
    std : np.ndarray of shape (n_features,) or None

    Returns
    -------
    X_scaled : np.ndarray of shape (n_samples, n_features)
    mean : np.ndarray of shape (n_features,)
    std : np.ndarray of shape (n_features,)
    """
    raise NotImplementedError


def create_polynomial_features(
    X: np.ndarray, degree: int = 2
) -> np.ndarray:
    """Create polynomial features up to the given degree.

    For degree=2 with features [x1, x2], output columns should be:
    [x1, x2, x1^2, x1*x2, x2^2]

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    degree : int, polynomial degree (only degree=2 is required)

    Returns
    -------
    np.ndarray of shape (n_samples, n_output_features)
        Includes original features plus polynomial combinations.
        Does NOT include a bias/intercept column.
    """
    raise NotImplementedError


def select_top_k_features(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    task: str = "classification",
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the top K features based on mutual information.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    k : int, number of features to select
    task : str, "classification" or "regression"
    random_state : int

    Returns
    -------
    X_selected : np.ndarray of shape (n_samples, k)
    selected_indices : np.ndarray of shape (k,), indices of selected features
    """
    raise NotImplementedError


def build_feature_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    categorical_cols: list[int] = None,
    ordinal_cols: list[int] = None,
    ordinal_mappings: dict[int, dict] = None,
    k_features: int = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a feature engineering pipeline: encode, scale, select.

    Steps:
    1. Ordinal-encode specified columns (if any).
    2. One-hot encode specified categorical columns (if any).
    3. Scale all features (fit on train, transform both).
    4. Select top K features (if k_features is not None).

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    X_test  : np.ndarray of shape (n_test, n_features)
    y_train : np.ndarray of shape (n_train,)
    categorical_cols : list[int] or None, column indices for one-hot encoding
    ordinal_cols : list[int] or None, column indices for ordinal encoding
    ordinal_mappings : dict mapping column index to category->int mapping
    k_features : int or None, number of features to select

    Returns
    -------
    X_train_processed : np.ndarray
    X_test_processed  : np.ndarray
    """
    raise NotImplementedError
