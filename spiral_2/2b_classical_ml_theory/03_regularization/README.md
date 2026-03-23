# Module 03: Regularization Theory

> Regularization is not a hack -- it is the MAP estimate under a prior. L2 = Gaussian
> prior. L1 = Laplace prior. This module builds the geometric and probabilistic
> intuition for why regularization works.

## Learning Objectives

- Derive ridge regression as MAP with Gaussian prior.
- Derive the soft-thresholding operator for L1 (lasso).
- Implement coordinate descent for lasso.
- Understand why L1 gives sparsity geometrically (diamond vs circle).
- Implement cross-validation for selecting regularization strength.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Bishop, *PRML*, section 3.1.4 | Regularized least squares |
| Murphy, *MLAPP*, section 7.5, 13.3 | Ridge and L1 |
| Hastie, Tibshirani, Friedman, *ESL*, ch 3.4 (free PDF) | The Lasso |

## Derive It

1. **Ridge as MAP.** Start with a Gaussian prior N(0, (1/alpha)I) on w. Show the
   MAP objective is ||y-Xw||^2 + alpha||w||^2. The closed-form solution is
   w = (X^T X + alpha I)^{-1} X^T y.

2. **Soft-thresholding.** For the 1D lasso problem min_w (y-w)^2 + alpha|w|,
   derive the solution: w* = sign(y) max(|y| - alpha/2, 0). This is the
   proximal operator for L1.

3. **Geometric intuition.** In 2D weight space, draw the contours of the
   unregularized loss. Overlay the L1 diamond and L2 circle. The constrained
   optimum is where they first touch. The L1 diamond has corners on the axes,
   which is why L1 solutions hit the axes (sparsity).

## "Naive then Derive" Challenge

Visualize L1 vs L2 constraint regions in 2D. Show geometrically why L1 gives
sparsity. Then verify on synthetic data with 5 true features and 95 noise features.

## Exercises

See `exercises.py`.
