"""
Tests for Module 06: Mini-Project — Customer Segmentation

Run: pytest tests/test_mini_project.py -v
"""

import numpy as np
import pytest

from mini_project import segment_customers


class TestSegmentCustomers:
    def setup_method(self):
        rng = np.random.RandomState(42)
        # Simulate customer data: 3 natural clusters with different scales
        # Feature 0: income (large scale)
        # Feature 1: age (small scale)
        # Feature 2: spending score (medium scale)
        c1 = np.column_stack([
            rng.normal(30000, 5000, 50),
            rng.normal(25, 3, 50),
            rng.normal(70, 10, 50),
        ])
        c2 = np.column_stack([
            rng.normal(80000, 10000, 50),
            rng.normal(45, 5, 50),
            rng.normal(40, 8, 50),
        ])
        c3 = np.column_stack([
            rng.normal(120000, 15000, 50),
            rng.normal(55, 4, 50),
            rng.normal(60, 12, 50),
        ])
        self.X = np.vstack([c1, c2, c3])

    def test_returns_dict(self):
        result = segment_customers(self.X)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = segment_customers(self.X)
        for key in ["labels", "centroids", "chosen_k", "inertias",
                     "pca_coords", "explained_variance"]:
            assert key in result, f"Missing key: {key}"

    def test_labels_shape(self):
        result = segment_customers(self.X)
        assert result["labels"].shape == (150,)

    def test_pca_coords_shape(self):
        result = segment_customers(self.X)
        assert result["pca_coords"].shape == (150, 2)

    def test_explained_variance_shape(self):
        result = segment_customers(self.X)
        assert result["explained_variance"].shape == (2,)

    def test_inertias_length(self):
        result = segment_customers(self.X, max_k=8)
        assert len(result["inertias"]) == 8

    def test_chosen_k_reasonable(self):
        result = segment_customers(self.X)
        # With 3 natural clusters, chosen K should be 2-5
        assert 2 <= result["chosen_k"] <= 5

    def test_centroids_shape_matches_k(self):
        result = segment_customers(self.X)
        k = result["chosen_k"]
        assert result["centroids"].shape[0] == k
        assert result["centroids"].shape[1] == 3  # n_features

    def test_labels_match_chosen_k(self):
        result = segment_customers(self.X)
        assert len(np.unique(result["labels"])) == result["chosen_k"]
