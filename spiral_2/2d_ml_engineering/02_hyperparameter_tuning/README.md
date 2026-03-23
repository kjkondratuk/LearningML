# Module 02: Hyperparameter Tuning

> Grid search wastes evaluations on unimportant parameters. Random search is
> surprisingly competitive. Bayesian optimization is optimal. This module builds
> all three from scratch.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Bergstra & Bengio, "Random Search for Hyper-Parameter Optimization" (2012) | Why random beats grid |
| Snoek et al., "Practical Bayesian Optimization of ML Algorithms" (2012) | BO derivation |
| Optuna documentation | Production-quality HPO tool |

## Derive It

1. **Random vs grid in high dimensions.** With 5 parameters and 10 values each,
   grid search requires 100,000 evaluations. Random search with 100 evaluations
   finds a comparable result because most parameters are unimportant -- random
   search explores more values of the important ones.

2. **Expected Improvement.** Derive EI(x) = E[max(f(x) - f_best, 0)] under the
   GP posterior. Show it balances exploitation (high predicted mean) and
   exploration (high uncertainty).

## Exercises

See `exercises.py`.

## Mini-Project: Bayesian Optimization from Scratch

GP surrogate, EI acquisition, full BO loop with visualization. See `mini_project.py`.
