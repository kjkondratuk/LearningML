# Module 07: Dimensionality Reduction

> Most data lives on low-dimensional manifolds embedded in high-dimensional space.
> PCA finds linear subspaces; kernel PCA and t-SNE find non-linear manifolds.

## Learning Objectives

- Implement PCA via eigendecomposition and SVD, show equivalence.
- Understand explained variance and the scree plot.
- Implement kernel PCA for non-linear dimensionality reduction.
- Implement t-SNE from scratch (the hardest exercise in this phase).
- Implement PCA whitening and understand its purpose.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Bishop, *PRML*, section 12.1 | PCA derivation |
| Murphy, *MLAPP*, ch 12 | Latent linear models |
| Van der Maaten & Hinton, "Visualizing Data using t-SNE" (2008) | Sections 1--3 |
| 3Blue1Brown, "Eigenvectors and Eigenvalues" | PCA intuition |

## Derive It

1. **PCA as maximum variance projection.** Show that the first principal component
   direction w_1 maximizes Var(w_1^T X) = w_1^T C w_1 subject to ||w_1|| = 1.
   This is the eigenvector of C with the largest eigenvalue.

2. **PCA via SVD.** Show that if X = U S V^T (centered), then the principal
   components are V and the projections are U S.

3. **t-SNE gradient.** Derive the gradient of KL(P || Q) where P is the
   high-dimensional pairwise affinity (Gaussian) and Q is the low-dimensional
   pairwise affinity (Student-t). The Student-t in the low-D space creates
   heavier tails, solving the "crowding problem."

## "Naive then Derive" Challenge

Run PCA on the Swiss roll dataset. Show it fails (linear projection cannot
unroll the manifold). Then implement kernel PCA with an RBF kernel and show
it finds the unrolled structure.

## Exercises

See `exercises.py`.

## Mini-Project: PCA + t-SNE on Fashion-MNIST

Compare linear (PCA) and non-linear (t-SNE) dimensionality reduction on
Fashion-MNIST. See `mini_project.py`.
