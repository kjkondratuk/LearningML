# Module 01: Linear Algebra

> Every ML algorithm is a sequence of matrix operations in disguise. This module builds
> the decompositions you will use throughout the rest of the curriculum.

## Learning Objectives

- Implement matrix multiplication from nested loops, then understand why NumPy exists.
- Implement LU, QR, eigendecomposition, and SVD from scratch.
- Understand why SVD is the Swiss Army knife of ML (pseudoinverse, PCA, low-rank
  approximation, matrix completion).
- Build fluency reading matrix notation in papers.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| 3Blue1Brown, *Essence of Linear Algebra* (full series, ch 1--15) | Geometric intuition for every operation |
| Strang, *Introduction to Linear Algebra*, chapters 1--4, 6--7 | Rigorous treatment of decompositions |
| Bishop, *PRML*, Appendix C | Matrix identities you will use repeatedly |

## Derive It (pen-and-paper before code)

1. **LU Decomposition.** Starting from Gaussian elimination, show that every step can
   be captured as a lower-triangular matrix L such that A = LU.

2. **QR via Gram-Schmidt.** Given columns a_1, ..., a_n of A, derive the orthogonal
   vectors q_1, ..., q_n and the upper-triangular coefficients R.

3. **Eigendecomposition.** Prove that if Av = lambda v, then A^k v = lambda^k v. This
   is why the power iteration method works -- the dominant eigenvalue dominates
   exponentially.

4. **SVD from eigendecomposition.** Show that the left singular vectors of A are the
   eigenvectors of AA^T, the right singular vectors are the eigenvectors of A^T A,
   and the singular values are the square roots of the shared eigenvalues.

5. **Pseudoinverse via SVD.** Given A = U S V^T, derive A^+ = V S^+ U^T where S^+ is
   formed by reciprocating the non-zero singular values. Show that x = A^+ b is the
   least-squares solution to Ax = b.

## "Naive then Derive" Challenge

Implement `matrix_multiply` with three nested Python for-loops. Time it on 500x500
matrices. Then implement the same computation with `np.dot`. The difference -- typically
100x or more -- demonstrates why NumPy (and BLAS/LAPACK underneath it) exist. Every
time you are tempted to write a for-loop over matrix elements, remember this exercise.

## Exercises

See `exercises.py` for stubs. Each function has a docstring stating precisely what to
implement and what mathematical result to derive first.

## Mini-Project: Image Compression via SVD

Load a grayscale image, compute its SVD, and reconstruct it using only the top-k
singular values. Explore the trade-off between compression ratio and reconstruction
quality. See `mini_project.py`.

## Style Notes

- All functions accept and return NumPy arrays.
- Use `np.allclose` with tolerance 1e-10 for reconstruction checks.
- Document which decomposition each function relies on in its docstring.
