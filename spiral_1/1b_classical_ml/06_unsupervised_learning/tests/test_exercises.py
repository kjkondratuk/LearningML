"""
Tests for Module 06: Unsupervised Learning

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import pytest

from exercises import (
    kmeans_assign,
    kmeans_update,
    kmeans_from_scratch,
    find_optimal_k,
    sklearn_pca,
    sklearn_hierarchical,
)


# ---------------------------------------------------------------------------
# kmeans_assign
# ---------------------------------------------------------------------------

class TestKMeansAssign:
    def test_simple(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]])
        centroids = np.array([[0.5, 0.5], [10.5, 10.5]])
        labels = kmeans_assign(X, centroids)
        np.testing.assert_array_equal(labels, [0, 0, 1, 1])

    def test_output_shape(self):
        X = np.random.RandomState(42).randn(50, 3)
        centroids = np.random.RandomState(0).randn(4, 3)
        labels = kmeans_assign(X, centroids)
        assert labels.shape == (50,)

    def test_all_labels_valid(self):
        X = np.random.RandomState(42).randn(30, 2)
        centroids = np.random.RandomState(0).randn(3, 2)
        labels = kmeans_assign(X, centroids)
        assert np.all(labels >= 0)
        assert np.all(labels < 3)

    def test_single_centroid(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        centroids = np.array([[0.0, 0.0]])
        labels = kmeans_assign(X, centroids)
        np.testing.assert_array_equal(labels, [0, 0])


# ---------------------------------------------------------------------------
# kmeans_update
# ---------------------------------------------------------------------------

class TestKMeansUpdate:
    def test_simple(self):
        X = np.array([[0.0, 0.0], [2.0, 2.0], [10.0, 10.0], [12.0, 12.0]])
        labels = np.array([0, 0, 1, 1])
        centroids = kmeans_update(X, labels, 2)
        np.testing.assert_allclose(centroids[0], [1.0, 1.0])
        np.testing.assert_allclose(centroids[1], [11.0, 11.0])

    def test_output_shape(self):
        X = np.random.RandomState(42).randn(20, 4)
        labels = np.random.RandomState(0).randint(0, 3, 20)
        centroids = kmeans_update(X, labels, 3)
        assert centroids.shape == (3, 4)

    def test_single_point_cluster(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        labels = np.array([0, 1, 1])
        centroids = kmeans_update(X, labels, 2)
        np.testing.assert_allclose(centroids[0], [1.0, 2.0])
        np.testing.assert_allclose(centroids[1], [4.0, 5.0])


# ---------------------------------------------------------------------------
# kmeans_from_scratch
# ---------------------------------------------------------------------------

class TestKMeansFromScratch:
    def setup_method(self):
        rng = np.random.RandomState(42)
        # Three well-separated clusters
        c1 = rng.randn(30, 2) + np.array([0, 0])
        c2 = rng.randn(30, 2) + np.array([10, 0])
        c3 = rng.randn(30, 2) + np.array([5, 10])
        self.X = np.vstack([c1, c2, c3])

    def test_finds_three_clusters(self):
        labels, centroids, n_iters = kmeans_from_scratch(self.X, k=3)
        assert len(np.unique(labels)) == 3

    def test_output_shapes(self):
        labels, centroids, n_iters = kmeans_from_scratch(self.X, k=3)
        assert labels.shape == (90,)
        assert centroids.shape == (3, 2)
        assert isinstance(n_iters, int)

    def test_converges(self):
        _, _, n_iters = kmeans_from_scratch(self.X, k=3, max_iter=100)
        assert n_iters < 100, "Should converge before max_iter on well-separated data"

    def test_reproducible(self):
        l1, c1, _ = kmeans_from_scratch(self.X, k=3, random_state=7)
        l2, c2, _ = kmeans_from_scratch(self.X, k=3, random_state=7)
        np.testing.assert_array_equal(l1, l2)
        np.testing.assert_array_equal(c1, c2)

    def test_labels_valid(self):
        labels, _, _ = kmeans_from_scratch(self.X, k=5)
        assert np.all(labels >= 0)
        assert np.all(labels < 5)


# ---------------------------------------------------------------------------
# find_optimal_k
# ---------------------------------------------------------------------------

class TestFindOptimalK:
    def test_returns_list(self):
        X = np.random.RandomState(42).randn(50, 2)
        inertias = find_optimal_k(X, max_k=5)
        assert isinstance(inertias, list)
        assert len(inertias) == 5

    def test_inertia_decreases(self):
        rng = np.random.RandomState(42)
        c1 = rng.randn(50, 2)
        c2 = rng.randn(50, 2) + 10
        X = np.vstack([c1, c2])
        inertias = find_optimal_k(X, max_k=5)
        # Inertia should generally decrease as K increases
        assert inertias[0] > inertias[-1]

    def test_k1_inertia_is_total_variance(self):
        X = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])
        inertias = find_optimal_k(X, max_k=3)
        # K=1: centroid is [2, 0], inertia = 4 + 0 + 4 = 8
        assert inertias[0] == pytest.approx(8.0, abs=0.5)

    def test_all_non_negative(self):
        X = np.random.RandomState(0).randn(30, 2)
        inertias = find_optimal_k(X, max_k=5)
        assert all(i >= 0 for i in inertias)


# ---------------------------------------------------------------------------
# sklearn_pca
# ---------------------------------------------------------------------------

class TestSklearnPCA:
    def test_output_shape(self):
        X = np.random.RandomState(42).randn(50, 10)
        X_reduced, explained = sklearn_pca(X, n_components=3)
        assert X_reduced.shape == (50, 3)
        assert explained.shape == (3,)

    def test_default_2d(self):
        X = np.random.RandomState(42).randn(20, 5)
        X_reduced, explained = sklearn_pca(X)
        assert X_reduced.shape == (20, 2)

    def test_variance_sums_to_less_than_one(self):
        X = np.random.RandomState(42).randn(50, 10)
        _, explained = sklearn_pca(X, n_components=3)
        assert np.sum(explained) <= 1.0 + 1e-10

    def test_variance_is_sorted(self):
        X = np.random.RandomState(42).randn(50, 10)
        _, explained = sklearn_pca(X, n_components=5)
        # Each component should explain >= the next one
        for i in range(len(explained) - 1):
            assert explained[i] >= explained[i + 1] - 1e-10


# ---------------------------------------------------------------------------
# sklearn_hierarchical
# ---------------------------------------------------------------------------

class TestSklearnHierarchical:
    def test_output_shape(self):
        X = np.random.RandomState(42).randn(30, 2)
        labels = sklearn_hierarchical(X, n_clusters=3)
        assert labels.shape == (30,)

    def test_correct_number_of_clusters(self):
        X = np.random.RandomState(42).randn(30, 2)
        labels = sklearn_hierarchical(X, n_clusters=4)
        assert len(np.unique(labels)) == 4

    def test_finds_obvious_clusters(self):
        rng = np.random.RandomState(42)
        c1 = rng.randn(20, 2) + np.array([0, 0])
        c2 = rng.randn(20, 2) + np.array([20, 20])
        X = np.vstack([c1, c2])
        labels = sklearn_hierarchical(X, n_clusters=2)
        # All points in c1 should have the same label, and c2 should have the other
        assert len(np.unique(labels[:20])) == 1
        assert len(np.unique(labels[20:])) == 1
        assert labels[0] != labels[20]
