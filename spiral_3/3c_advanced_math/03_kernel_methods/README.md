# Module 03: Kernel Methods & RKHS

## Learning Objectives

- Implement RBF and polynomial kernels and verify Mercer conditions
- Build kernel ridge regression and kernel PCA
- Understand the connection between kernels and Gaussian processes
- Implement Nystrom approximation for scalable kernel methods

## Key References

- **Scholkopf & Smola.** "Learning with Kernels." MIT Press, 2002. Ch. 2.
- **Rasmussen & Williams.** "Gaussian Processes for Machine Learning." Ch. 2.

## Theoretical Background

A kernel k(x,y) is a function such that the Gram matrix K_ij = k(x_i, x_j) is positive
semi-definite for any set of points. By Mercer's theorem, such a kernel corresponds to
an inner product in some (possibly infinite-dimensional) feature space.

The representer theorem guarantees that the solution to any regularized risk minimization
in an RKHS lies in the span of the kernel evaluations at the training points.

> **Math Callout:** Hilbert spaces, positive definite functions, Mercer's theorem,
> representer theorem. Resource: Scholkopf & Smola "Learning with Kernels."

## Exercises

1. `rbf_kernel` -- Gaussian RBF kernel
2. `polynomial_kernel` -- Polynomial kernel
3. `verify_mercer_conditions` -- Check PSD property
4. `kernel_ridge_regression` -- alpha = (K + lambda I)^{-1} y
5. `kernel_pca` -- PCA in feature space
6. `nystrom_approximation` -- Low-rank kernel approximation
