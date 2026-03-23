# Module 04: Bias-Variance Tradeoff

> The bias-variance decomposition is the theoretical backbone of model selection.
> Derive it formally, implement it empirically, then discover that double descent
> breaks the simple U-curve intuition.

## Learning Objectives

- Derive the bias-variance decomposition: E[error] = bias^2 + variance + noise.
- Implement the decomposition empirically using repeated training sets.
- Produce the classic U-shaped test error curve.
- Discover double descent for overparameterized models.
- Diagnose high bias vs high variance from learning curves.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Bishop, *PRML*, section 3.2 | The bias-variance decomposition derivation |
| Murphy, *MLAPP*, section 6.4 | Bias-variance tradeoff |
| Hastie, *ESL*, ch 7.2--7.3 | Estimation of prediction error |

## Derive It

1. **The decomposition.** For a fixed test point x, show:
   E[(y - f_hat(x))^2] = (E[f_hat(x)] - f(x))^2 + Var[f_hat(x)] + sigma^2
   = bias^2 + variance + irreducible_noise

2. **Polynomial example.** For polynomial regression of degree d:
   - Low d (1): high bias (underfitting), low variance
   - High d (20): low bias, high variance (overfitting)
   - Optimal d: minimizes the sum

## "Naive then Derive" Challenge

Your intuition from Spiral 1 is "more complex = overfitting." Now demonstrate
double descent: for sufficiently overparameterized models, test error decreases
again past the interpolation threshold.

## Exercises

See `exercises.py`.
