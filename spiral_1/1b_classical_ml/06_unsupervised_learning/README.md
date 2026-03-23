# Module 06: Unsupervised Learning

## Overview

Unsupervised learning finds structure in data **without labels**. You have inputs
but no "right answers." This module covers the three pillars: clustering (K-means),
dimensionality reduction (PCA), and hierarchical clustering.

## Core Concepts

### K-Means Clustering

K-means partitions data into K clusters by iterating two steps:

1. **Assign** each point to the nearest centroid.
2. **Update** each centroid to the mean of its assigned points.

Repeat until centroids stop moving (or a max iteration count).

**Go parallel:** K-means is like a load balancer. You have K servers (centroids)
and N requests (data points). Each request goes to the closest server. Then each
server moves to the center of its assigned requests. Repeat. Eventually the system
stabilizes.

### Choosing K (Elbow Method)

There is no free lunch — you need to pick K. The elbow method plots the sum of
squared distances (inertia) vs. K and looks for the "elbow" where adding more
clusters stops helping much.

### PCA (Principal Component Analysis)

PCA finds the directions of maximum variance in the data. It projects
high-dimensional data onto fewer dimensions while preserving as much information
as possible.

**Math callout:**

PCA computes the eigenvectors of the covariance matrix. The eigenvector with the
largest eigenvalue is the first principal component (direction of most variance).

You do not need to implement eigendecomposition. Use sklearn. But understand that
PCA is a linear transformation that rotates your data so the axes align with
variance.

**Go parallel:** PCA is like compressing a large struct into a smaller one that
captures the most important fields. You lose some information, but the compressed
version is much easier to work with.

### Hierarchical Clustering

Builds a tree (dendrogram) of clusters by iteratively merging the two closest
clusters. Unlike K-means, you do not need to specify K upfront — you choose where
to "cut" the tree.

Resources:
- [StatQuest: K-Means](https://www.youtube.com/watch?v=4b5d3muPQmA)
- [StatQuest: PCA](https://www.youtube.com/watch?v=FgakZw6K1QQ)
- [Scikit-learn Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

## Exercises

Implement in `exercises.py`:

1. `kmeans_assign` — Assign each point to the nearest centroid.
2. `kmeans_update` — Recompute centroids from assignments.
3. `kmeans_from_scratch` — Full K-means algorithm using assign + update.
4. `find_optimal_k` — Compute inertia for K=1..max_k, return the inertias.
5. `sklearn_pca` — Reduce dimensionality using sklearn PCA.
6. `sklearn_hierarchical` — Cluster using sklearn AgglomerativeClustering.

## Mini-Project

In `mini_project.py`:

`segment_customers` — Given customer feature data, standardize it, find optimal K,
cluster with K-means, and reduce to 2D with PCA for visualization. Return cluster
labels and 2D coordinates.

## Do It the Wrong Way (on purpose)

Before implementing `segment_customers` correctly, try clustering **without
normalizing** the features. If one feature is income in [20000, 200000] and another
is age in [18, 80], K-means will be dominated by income because distances in that
dimension are so much larger. The clusters will be essentially 1D slices of income.

After you see the bad results, add standardization and watch the clusters become
meaningful across all features.

## Style Notes

- For `kmeans_assign`, use vectorized distance computation (broadcasting or
  `np.linalg.norm`), not Python loops over points.
- `kmeans_from_scratch` should accept a `max_iter` parameter and stop early if
  centroids do not change.
- Use `np.random.RandomState` for centroid initialization to ensure reproducibility.
