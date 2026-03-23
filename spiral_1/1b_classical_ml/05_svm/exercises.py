"""
Module 05: Support Vector Machines — Exercise Stubs

Implement each function until `pytest tests/test_exercises.py` passes.
"""

import numpy as np


def sklearn_svm_linear(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    C: float = 1.0,
    random_state: int = 42,
) -> np.ndarray:
    """Train a linear SVM and return predictions on the test set.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    y_train : np.ndarray of shape (n_train,)
    X_test  : np.ndarray of shape (n_test, n_features)
    C : float, regularization parameter
    random_state : int

    Returns
    -------
    np.ndarray of shape (n_test,), predicted labels
    """
    raise NotImplementedError


def sklearn_svm_rbf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    C: float = 1.0,
    gamma: str | float = "scale",
    random_state: int = 42,
) -> np.ndarray:
    """Train an RBF SVM and return predictions on the test set.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    y_train : np.ndarray of shape (n_train,)
    X_test  : np.ndarray of shape (n_test, n_features)
    C : float
    gamma : str or float
    random_state : int

    Returns
    -------
    np.ndarray of shape (n_test,), predicted labels
    """
    raise NotImplementedError


def compare_kernels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> dict[str, float]:
    """Train SVMs with linear, rbf, and poly kernels; return test accuracy for each.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    y_train : np.ndarray of shape (n_train,)
    X_test  : np.ndarray of shape (n_test, n_features)
    y_test  : np.ndarray of shape (n_test,)
    random_state : int

    Returns
    -------
    dict mapping kernel name ("linear", "rbf", "poly") to test accuracy (float)
    """
    raise NotImplementedError


def visualize_decision_boundary(
    X: np.ndarray,
    y: np.ndarray,
    kernel: str = "rbf",
    C: float = 1.0,
    resolution: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a decision boundary mesh for a 2D dataset.

    Train an SVM on X, y, then predict on a grid covering the feature space.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, 2)
    y : np.ndarray of shape (n_samples,)
    kernel : str
    C : float
    resolution : float, step size for the meshgrid
    random_state : int

    Returns
    -------
    xx : np.ndarray, meshgrid x-coordinates
    yy : np.ndarray, meshgrid y-coordinates
    Z  : np.ndarray, predicted class for each grid point (same shape as xx)
    """
    raise NotImplementedError


def grid_search_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> dict:
    """Use GridSearchCV to find optimal C and gamma for an RBF SVM.

    Search over C in [0.1, 1, 10, 100] and gamma in [0.001, 0.01, 0.1, 1].
    Use 5-fold cross-validation.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    y_train : np.ndarray of shape (n_train,)
    X_test  : np.ndarray of shape (n_test, n_features)
    y_test  : np.ndarray of shape (n_test,)
    random_state : int

    Returns
    -------
    dict with keys:
        "best_C"         : float
        "best_gamma"     : float
        "best_cv_score"  : float, best cross-validation accuracy
        "test_accuracy"  : float, test set accuracy with best params
    """
    raise NotImplementedError
