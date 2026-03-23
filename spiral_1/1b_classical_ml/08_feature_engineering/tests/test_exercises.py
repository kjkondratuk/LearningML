"""
Tests for Module 08: Feature Engineering

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    encode_ordinal,
    encode_onehot_manual,
    scale_features,
    create_polynomial_features,
    select_top_k_features,
    build_feature_pipeline,
)


# ---------------------------------------------------------------------------
# encode_ordinal
# ---------------------------------------------------------------------------

class TestEncodeOrdinal:
    def test_simple(self):
        values = np.array(["low", "medium", "high", "low"])
        mapping = {"low": 0, "medium": 1, "high": 2}
        result = encode_ordinal(values, mapping)
        np.testing.assert_array_equal(result, [0, 1, 2, 0])

    def test_output_type(self):
        values = np.array(["a", "b"])
        mapping = {"a": 0, "b": 1}
        result = encode_ordinal(values, mapping)
        assert isinstance(result, np.ndarray)
        assert result.dtype in [np.int64, np.int32, int]

    def test_preserves_order(self):
        values = np.array(["cold", "warm", "hot", "warm", "cold"])
        mapping = {"cold": 0, "warm": 1, "hot": 2}
        result = encode_ordinal(values, mapping)
        np.testing.assert_array_equal(result, [0, 1, 2, 1, 0])


# ---------------------------------------------------------------------------
# encode_onehot_manual
# ---------------------------------------------------------------------------

class TestEncodeOneHot:
    def test_simple(self):
        values = np.array(["cat", "dog", "cat", "fish"])
        encoded, categories = encode_onehot_manual(values)
        assert encoded.shape == (4, 3)
        assert sorted(categories) == categories  # Should be sorted

    def test_binary(self):
        values = np.array(["yes", "no", "yes"])
        encoded, categories = encode_onehot_manual(values)
        assert encoded.shape == (3, 2)
        # Each row should sum to 1
        np.testing.assert_array_equal(encoded.sum(axis=1), [1, 1, 1])

    def test_values_are_binary(self):
        values = np.array(["a", "b", "c", "a"])
        encoded, _ = encode_onehot_manual(values)
        assert set(np.unique(encoded)) == {0, 1}

    def test_correct_encoding(self):
        values = np.array(["b", "a", "c"])
        encoded, categories = encode_onehot_manual(values)
        # Categories sorted: ["a", "b", "c"]
        assert categories == ["a", "b", "c"]
        # "b" -> [0, 1, 0], "a" -> [1, 0, 0], "c" -> [0, 0, 1]
        expected = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        np.testing.assert_array_equal(encoded, expected)

    def test_single_category(self):
        values = np.array(["x", "x", "x"])
        encoded, categories = encode_onehot_manual(values)
        assert encoded.shape == (3, 1)
        np.testing.assert_array_equal(encoded, [[1], [1], [1]])


# ---------------------------------------------------------------------------
# scale_features
# ---------------------------------------------------------------------------

class TestScaleFeatures:
    def test_zero_mean_unit_var(self):
        X = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        X_scaled, mean, std = scale_features(X)
        np.testing.assert_allclose(X_scaled.mean(axis=0), [0, 0], atol=1e-10)
        np.testing.assert_allclose(X_scaled.std(axis=0), [1, 1], atol=1e-10)

    def test_returns_params(self):
        X = np.array([[1.0], [3.0], [5.0]])
        _, mean, std = scale_features(X)
        assert mean.shape == (1,)
        assert std.shape == (1,)
        assert mean[0] == pytest.approx(3.0)

    def test_apply_to_test_data(self):
        X_train = np.array([[1.0, 10.0], [3.0, 30.0]])
        _, mean, std = scale_features(X_train)
        X_test = np.array([[2.0, 20.0]])
        X_test_scaled, _, _ = scale_features(X_test, mean=mean, std=std)
        # (2 - 2) / 1 = 0, (20 - 20) / 10 = 0
        np.testing.assert_allclose(X_test_scaled, [[0.0, 0.0]], atol=1e-10)

    def test_handles_zero_variance(self):
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        X_scaled, _, std = scale_features(X)
        # Column with zero variance should not produce inf or nan
        assert not np.any(np.isnan(X_scaled))
        assert not np.any(np.isinf(X_scaled))


# ---------------------------------------------------------------------------
# create_polynomial_features
# ---------------------------------------------------------------------------

class TestPolynomialFeatures:
    def test_single_feature(self):
        X = np.array([[2.0], [3.0]])
        result = create_polynomial_features(X, degree=2)
        # [x1, x1^2] = [2, 4], [3, 9]
        expected = np.array([[2.0, 4.0], [3.0, 9.0]])
        np.testing.assert_allclose(result, expected)

    def test_two_features(self):
        X = np.array([[1.0, 2.0]])
        result = create_polynomial_features(X, degree=2)
        # [x1, x2, x1^2, x1*x2, x2^2]
        expected = np.array([[1.0, 2.0, 1.0, 2.0, 4.0]])
        np.testing.assert_allclose(result, expected)

    def test_output_shape_two_features(self):
        X = np.random.RandomState(42).randn(10, 2)
        result = create_polynomial_features(X, degree=2)
        # 2 original + 3 degree-2 (x1^2, x1*x2, x2^2) = 5
        assert result.shape == (10, 5)

    def test_output_shape_three_features(self):
        X = np.random.RandomState(42).randn(10, 3)
        result = create_polynomial_features(X, degree=2)
        # 3 original + 6 degree-2 (x1^2, x1*x2, x1*x3, x2^2, x2*x3, x3^2) = 9
        assert result.shape == (10, 9)

    def test_includes_original(self):
        X = np.array([[2.0, 3.0]])
        result = create_polynomial_features(X, degree=2)
        # First columns should be the original features
        np.testing.assert_allclose(result[0, :2], [2.0, 3.0])


# ---------------------------------------------------------------------------
# select_top_k_features
# ---------------------------------------------------------------------------

class TestSelectTopKFeatures:
    def test_selects_k_features(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        X_sel, indices = select_top_k_features(X, y, k=3)
        assert X_sel.shape == (100, 3)
        assert indices.shape == (3,)

    def test_selected_columns_match(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = (X[:, 0] > 0).astype(int)
        X_sel, indices = select_top_k_features(X, y, k=2)
        np.testing.assert_array_equal(X_sel, X[:, indices])

    def test_informative_features_selected(self):
        rng = np.random.RandomState(42)
        # Feature 0 is informative, features 1-4 are noise
        X = np.column_stack([
            rng.randn(200),
            rng.randn(200) * 0.001,
            rng.randn(200) * 0.001,
            rng.randn(200) * 0.001,
            rng.randn(200) * 0.001,
        ])
        y = (X[:, 0] > 0).astype(int)
        _, indices = select_top_k_features(X, y, k=1)
        assert 0 in indices

    def test_regression_task(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = 3 * X[:, 0] + rng.randn(100) * 0.1
        X_sel, indices = select_top_k_features(
            X, y, k=2, task="regression"
        )
        assert X_sel.shape == (100, 2)


# ---------------------------------------------------------------------------
# build_feature_pipeline
# ---------------------------------------------------------------------------

class TestBuildFeaturePipeline:
    def test_scaling_only(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, 3)
        X_test = rng.randn(10, 3)
        y_train = rng.randint(0, 2, 50)

        X_tr, X_te = build_feature_pipeline(X_train, X_test, y_train)
        assert X_tr.shape[0] == 50
        assert X_te.shape[0] == 10
        # After scaling, training data should be roughly zero-mean
        np.testing.assert_allclose(X_tr.mean(axis=0), 0, atol=0.1)

    def test_with_feature_selection(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 10)
        X_test = rng.randn(20, 10)
        y_train = (X_train[:, 0] > 0).astype(int)

        X_tr, X_te = build_feature_pipeline(
            X_train, X_test, y_train, k_features=3
        )
        assert X_tr.shape == (100, 3)
        assert X_te.shape == (20, 3)

    def test_output_types(self):
        rng = np.random.RandomState(42)
        X_train = rng.randn(30, 4)
        X_test = rng.randn(5, 4)
        y_train = rng.randint(0, 2, 30)

        X_tr, X_te = build_feature_pipeline(X_train, X_test, y_train)
        assert isinstance(X_tr, np.ndarray)
        assert isinstance(X_te, np.ndarray)
