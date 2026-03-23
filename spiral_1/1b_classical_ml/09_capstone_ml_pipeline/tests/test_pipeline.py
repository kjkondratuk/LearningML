"""
Tests for Module 09: Capstone — End-to-End ML Pipeline

Run: pytest tests/test_pipeline.py -v
"""

import numpy as np
import pytest

from pipeline import (
    load_and_explore,
    build_features,
    train_and_evaluate_models,
    select_best_model,
    final_evaluation,
)


# ---------------------------------------------------------------------------
# load_and_explore
# ---------------------------------------------------------------------------

class TestLoadAndExplore:
    def test_regression_returns_dict(self):
        result = load_and_explore("regression")
        assert isinstance(result, dict)

    def test_regression_required_keys(self):
        result = load_and_explore("regression")
        for key in ["X", "y", "n_samples", "n_features", "feature_names", "target_stats"]:
            assert key in result, f"Missing key: {key}"

    def test_regression_shapes(self):
        result = load_and_explore("regression")
        assert result["X"].shape[0] == result["n_samples"]
        assert result["X"].shape[1] == result["n_features"]
        assert result["y"].shape[0] == result["n_samples"]

    def test_regression_target_stats(self):
        result = load_and_explore("regression")
        for key in ["mean", "std", "min", "max"]:
            assert key in result["target_stats"]

    def test_classification_returns_dict(self):
        result = load_and_explore("classification")
        assert isinstance(result, dict)

    def test_classification_target_stats(self):
        result = load_and_explore("classification")
        assert "class_counts" in result["target_stats"]
        counts = result["target_stats"]["class_counts"]
        assert isinstance(counts, dict)
        assert len(counts) >= 2

    def test_feature_names_length(self):
        result = load_and_explore("regression")
        assert len(result["feature_names"]) == result["n_features"]


# ---------------------------------------------------------------------------
# build_features
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def setup_method(self):
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(100, 5)
        self.X_test = rng.randn(20, 5)
        self.feature_names = ["f0", "f1", "f2", "f3", "f4"]

    def test_returns_tuple(self):
        result = build_features(self.X_train, self.X_test, self.feature_names)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_creates_new_features(self):
        X_tr, X_te, names = build_features(
            self.X_train, self.X_test, self.feature_names
        )
        # Must have at least 5 more features than original
        assert X_tr.shape[1] >= 10, "Should create at least 5 new features"
        assert X_te.shape[1] == X_tr.shape[1]

    def test_feature_names_match(self):
        X_tr, _, names = build_features(
            self.X_train, self.X_test, self.feature_names
        )
        assert len(names) == X_tr.shape[1]

    def test_sample_counts_preserved(self):
        X_tr, X_te, _ = build_features(
            self.X_train, self.X_test, self.feature_names
        )
        assert X_tr.shape[0] == 100
        assert X_te.shape[0] == 20

    def test_no_nans(self):
        X_tr, X_te, _ = build_features(
            self.X_train, self.X_test, self.feature_names
        )
        assert not np.any(np.isnan(X_tr))
        assert not np.any(np.isnan(X_te))


# ---------------------------------------------------------------------------
# train_and_evaluate_models
# ---------------------------------------------------------------------------

class TestTrainAndEvaluateModels:
    def setup_method(self):
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(100, 5)
        self.X_test = rng.randn(30, 5)

    def test_regression_at_least_four_models(self):
        y_train = 3 * self.X_train[:, 0] + np.random.RandomState(0).randn(100) * 0.5
        y_test = 3 * self.X_test[:, 0] + np.random.RandomState(1).randn(30) * 0.5
        results = train_and_evaluate_models(
            self.X_train, y_train, self.X_test, y_test, task="regression"
        )
        assert len(results) >= 4

    def test_classification_at_least_four_models(self):
        y_train = (self.X_train[:, 0] > 0).astype(int)
        y_test = (self.X_test[:, 0] > 0).astype(int)
        results = train_and_evaluate_models(
            self.X_train, y_train, self.X_test, y_test, task="classification"
        )
        assert len(results) >= 4

    def test_result_structure(self):
        y_train = 2 * self.X_train[:, 0]
        y_test = 2 * self.X_test[:, 0]
        results = train_and_evaluate_models(
            self.X_train, y_train, self.X_test, y_test, task="regression"
        )
        for r in results:
            assert "name" in r
            assert "train_score" in r
            assert "test_score" in r
            assert "hyperparameters" in r
            assert isinstance(r["name"], str)
            assert isinstance(r["hyperparameters"], dict)

    def test_distinct_model_names(self):
        y_train = (self.X_train[:, 0] > 0).astype(int)
        y_test = (self.X_test[:, 0] > 0).astype(int)
        results = train_and_evaluate_models(
            self.X_train, y_train, self.X_test, y_test, task="classification"
        )
        names = [r["name"] for r in results]
        assert len(names) == len(set(names)), "Model names should be unique"


# ---------------------------------------------------------------------------
# select_best_model
# ---------------------------------------------------------------------------

class TestSelectBestModel:
    def test_returns_dict(self):
        model_results = [
            {"name": "A", "train_score": 0.9, "test_score": 0.85, "hyperparameters": {}},
            {"name": "B", "train_score": 0.95, "test_score": 0.80, "hyperparameters": {}},
            {"name": "C", "train_score": 0.88, "test_score": 0.87, "hyperparameters": {}},
            {"name": "D", "train_score": 0.92, "test_score": 0.82, "hyperparameters": {}},
        ]
        result = select_best_model(model_results)
        assert isinstance(result, dict)

    def test_required_keys(self):
        model_results = [
            {"name": "A", "train_score": 0.9, "test_score": 0.85, "hyperparameters": {}},
            {"name": "B", "train_score": 0.95, "test_score": 0.90, "hyperparameters": {}},
        ]
        result = select_best_model(model_results)
        assert "best_model_name" in result
        assert "test_score" in result
        assert "explanation" in result

    def test_picks_best_test_score(self):
        model_results = [
            {"name": "A", "train_score": 0.99, "test_score": 0.70, "hyperparameters": {}},
            {"name": "B", "train_score": 0.85, "test_score": 0.84, "hyperparameters": {}},
        ]
        result = select_best_model(model_results)
        assert result["best_model_name"] == "B"
        assert result["test_score"] == pytest.approx(0.84)

    def test_explanation_length(self):
        model_results = [
            {"name": "A", "train_score": 0.9, "test_score": 0.85, "hyperparameters": {}},
            {"name": "B", "train_score": 0.88, "test_score": 0.86, "hyperparameters": {}},
            {"name": "C", "train_score": 0.92, "test_score": 0.83, "hyperparameters": {}},
            {"name": "D", "train_score": 0.87, "test_score": 0.82, "hyperparameters": {}},
        ]
        result = select_best_model(model_results)
        words = result["explanation"].split()
        assert len(words) >= 50, f"Explanation too short ({len(words)} words, need >= 50)"
        assert len(words) <= 300, f"Explanation too long ({len(words)} words, need <= 300)"


# ---------------------------------------------------------------------------
# final_evaluation
# ---------------------------------------------------------------------------

class TestFinalEvaluation:
    def setup_method(self):
        rng = np.random.RandomState(42)
        self.X_train = rng.randn(150, 5)
        self.X_test = rng.randn(50, 5)

    def test_returns_dict(self):
        y_train = 2 * self.X_train[:, 0] + rng_noise(150)
        y_test = 2 * self.X_test[:, 0] + rng_noise(50)
        result = final_evaluation(
            self.X_train, y_train, self.X_test, y_test, task="regression"
        )
        assert isinstance(result, dict)

    def test_overfitting_keys(self):
        y_train = (self.X_train[:, 0] > 0).astype(int)
        y_test = (self.X_test[:, 0] > 0).astype(int)
        result = final_evaluation(
            self.X_train, y_train, self.X_test, y_test, task="classification"
        )
        ov = result["overfitting"]
        assert "complexity_param" in ov
        assert "param_values" in ov
        assert "train_scores" in ov
        assert "test_scores" in ov
        assert len(ov["param_values"]) == len(ov["train_scores"])
        assert len(ov["param_values"]) == len(ov["test_scores"])

    def test_learning_curve_keys(self):
        y_train = (self.X_train[:, 0] > 0).astype(int)
        y_test = (self.X_test[:, 0] > 0).astype(int)
        result = final_evaluation(
            self.X_train, y_train, self.X_test, y_test, task="classification"
        )
        lc = result["learning_curve"]
        assert "train_sizes" in lc
        assert "train_scores" in lc
        assert "test_scores" in lc
        assert len(lc["train_sizes"]) == len(lc["train_scores"])

    def test_overfitting_multiple_points(self):
        y_train = (self.X_train[:, 0] > 0).astype(int)
        y_test = (self.X_test[:, 0] > 0).astype(int)
        result = final_evaluation(
            self.X_train, y_train, self.X_test, y_test, task="classification"
        )
        assert len(result["overfitting"]["param_values"]) >= 3

    def test_learning_curve_multiple_sizes(self):
        y_train = (self.X_train[:, 0] > 0).astype(int)
        y_test = (self.X_test[:, 0] > 0).astype(int)
        result = final_evaluation(
            self.X_train, y_train, self.X_test, y_test, task="classification"
        )
        assert len(result["learning_curve"]["train_sizes"]) >= 3


def rng_noise(n, seed=0):
    return np.random.RandomState(seed).randn(n) * 0.5
