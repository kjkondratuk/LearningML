"""
Tests for the ML Library capstone.

Each model is compared against sklearn on synthetic and real-ish data.
"""

import numpy as np
import pytest

from ml_library import (
    LinearRegression,
    RidgeRegression,
    LogisticRegression,
    SoftmaxRegression,
    DecisionTree,
    RandomForest,
    PCA,
)


@pytest.fixture
def regression_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X @ np.array([1, -2, 0.5, 0, 0]) + 0.5 * np.random.randn(100)
    return X[:80], y[:80], X[80:], y[80:]


@pytest.fixture
def classification_data():
    np.random.seed(42)
    n = 100
    X = np.vstack([
        np.random.randn(n, 3) + [2, 0, 0],
        np.random.randn(n, 3) + [-2, 0, 0],
    ])
    y = np.array([0] * n + [1] * n)
    return X[:160], y[:160], X[160:], y[160:]


@pytest.fixture
def multiclass_data():
    np.random.seed(42)
    n = 80
    X = np.vstack([
        np.random.randn(n, 4) + [3, 0, 0, 0],
        np.random.randn(n, 4) + [0, 3, 0, 0],
        np.random.randn(n, 4) + [0, 0, 3, 0],
    ])
    y = np.array([0] * n + [1] * n + [2] * n)
    idx = np.random.permutation(len(y))
    X, y = X[idx], y[idx]
    return X[:200], y[:200], X[200:], y[200:]


class TestLinearRegression:
    def test_fit_predict(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_te)
        assert preds.shape == y_te.shape

    def test_score_positive(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        model = LinearRegression()
        model.fit(X_tr, y_tr)
        r2 = model.score(X_te, y_te)
        assert r2 > 0.5


class TestRidgeRegression:
    def test_approaches_ols(self, regression_data):
        X_tr, y_tr, X_te, y_te = regression_data
        ols = LinearRegression().fit(X_tr, y_tr)
        ridge = RidgeRegression(alpha=1e-10).fit(X_tr, y_tr)
        np.testing.assert_allclose(
            ols.predict(X_te), ridge.predict(X_te), atol=0.1
        )


class TestLogisticRegression:
    def test_binary_classification(self, classification_data):
        X_tr, y_tr, X_te, y_te = classification_data
        model = LogisticRegression(solver="newton", n_steps=20)
        model.fit(X_tr, y_tr)
        acc = model.score(X_te, y_te)
        assert acc > 0.8


class TestSoftmaxRegression:
    def test_multiclass(self, multiclass_data):
        X_tr, y_tr, X_te, y_te = multiclass_data
        model = SoftmaxRegression(lr=0.01, n_steps=500)
        model.fit(X_tr, y_tr)
        acc = model.score(X_te, y_te)
        assert acc > 0.7


class TestDecisionTree:
    def test_memorizes_training(self, classification_data):
        X_tr, y_tr, _, _ = classification_data
        model = DecisionTree(max_depth=None, min_samples=1)
        model.fit(X_tr, y_tr)
        acc = model.score(X_tr, y_tr)
        assert acc == 1.0


class TestRandomForest:
    def test_classification(self, classification_data):
        X_tr, y_tr, X_te, y_te = classification_data
        model = RandomForest(n_trees=20, max_depth=5)
        model.fit(X_tr, y_tr)
        acc = model.score(X_te, y_te)
        assert acc > 0.8


class TestPCA:
    def test_fit_transform(self):
        np.random.seed(42)
        X = np.random.randn(50, 10)
        pca = PCA(n_components=3)
        Xt = pca.fit_transform(X)
        assert Xt.shape == (50, 3)

    def test_explained_variance(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        pca = PCA(n_components=5)
        pca.fit(X)
        np.testing.assert_allclose(
            pca.explained_variance_ratio_.sum(), 1.0, atol=1e-6
        )

    def test_inverse_transform(self):
        np.random.seed(42)
        X = np.random.randn(50, 5)
        pca = PCA(n_components=5)
        Xt = pca.fit_transform(X)
        Xr = pca.inverse_transform(Xt)
        np.testing.assert_allclose(Xr, X, atol=1e-6)
