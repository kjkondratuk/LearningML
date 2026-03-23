"""
Module 09: Capstone — End-to-End ML Pipeline

Build a complete ML pipeline from data loading to model selection.
"""

import numpy as np


def load_and_explore(task: str = "regression") -> dict:
    """Load a dataset and return exploratory statistics.

    For "regression", use a housing-style dataset (e.g., sklearn's
    fetch_california_housing or make_regression).
    For "classification", use a binary classification dataset (e.g.,
    sklearn's make_classification or load_breast_cancer).

    Parameters
    ----------
    task : str, "regression" or "classification"

    Returns
    -------
    dict with keys:
        "X"              : np.ndarray of shape (n_samples, n_features)
        "y"              : np.ndarray of shape (n_samples,)
        "n_samples"      : int
        "n_features"     : int
        "feature_names"  : list[str]
        "target_stats"   : dict with "mean", "std", "min", "max" for regression
                           or "class_counts" dict for classification
    """
    raise NotImplementedError


def build_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str] = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Engineer features from raw data.

    Must create at least 5 new features beyond the original columns.
    Apply the same transformations to both train and test.

    Parameters
    ----------
    X_train : np.ndarray of shape (n_train, n_features)
    X_test  : np.ndarray of shape (n_test, n_features)
    feature_names : list[str] or None, names of original features

    Returns
    -------
    X_train_engineered : np.ndarray
    X_test_engineered  : np.ndarray
    new_feature_names  : list[str], names of all features in output
    """
    raise NotImplementedError


def train_and_evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "regression",
    random_state: int = 42,
) -> list[dict]:
    """Train at least 4 models and evaluate each on train and test sets.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    X_test  : np.ndarray
    y_test  : np.ndarray
    task : str, "regression" or "classification"
    random_state : int

    Returns
    -------
    list of dicts, each with keys:
        "name"            : str, model name
        "train_score"     : float
        "test_score"      : float
        "hyperparameters" : dict
    """
    raise NotImplementedError


def select_best_model(model_results: list[dict]) -> dict:
    """Select the best model and provide justification.

    Parameters
    ----------
    model_results : list[dict], output from train_and_evaluate_models

    Returns
    -------
    dict with keys:
        "best_model_name" : str
        "test_score"      : float
        "explanation"     : str, 50-300 word paragraph explaining the choice
    """
    raise NotImplementedError


def final_evaluation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "regression",
    random_state: int = 42,
) -> dict:
    """Produce diagnostic data: overfitting plot and learning curve.

    Parameters
    ----------
    X_train : np.ndarray
    y_train : np.ndarray
    X_test  : np.ndarray
    y_test  : np.ndarray
    task : str
    random_state : int

    Returns
    -------
    dict with keys:
        "overfitting" : dict with keys:
            "complexity_param" : str, name of the parameter varied
            "param_values"     : list, values tried
            "train_scores"     : list[float]
            "test_scores"      : list[float]
        "learning_curve" : dict with keys:
            "train_sizes"  : list[int]
            "train_scores" : list[float]
            "test_scores"  : list[float]
    """
    raise NotImplementedError
