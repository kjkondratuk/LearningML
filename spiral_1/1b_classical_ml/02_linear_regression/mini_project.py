"""
Module 02: Mini-Project — Predict Housing Prices

Tie together your linear regression exercises into a realistic pipeline.
"""

import numpy as np


def predict_housing_prices(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """Build a complete linear regression pipeline for housing price prediction.

    Steps:
    1. Split data into train/test sets.
    2. Standardize features (fit on train, transform both).
    3. Train linear regression using your gradient descent implementation.
    4. Also train using sklearn for comparison.
    5. Evaluate both on the test set using MSE.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,)
    test_size : float
    random_state : int

    Returns
    -------
    dict with keys:
        "gd_mse"      : float, test MSE from gradient descent model
        "sklearn_mse"  : float, test MSE from sklearn model
        "gd_weights"   : np.ndarray, learned weights from gradient descent
        "sklearn_coef" : np.ndarray, learned coefficients from sklearn
        "loss_history"  : list[float], training loss at each GD iteration
    """
    raise NotImplementedError
