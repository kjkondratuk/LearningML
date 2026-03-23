# Module 05: Scaling Laws & In-Context Learning

## Learning Objectives

- Fit power law scaling curves to compute-loss data
- Compute Chinchilla-optimal model/data ratios
- Understand in-context learning as implicit Bayesian inference
- Detect and analyze emergent abilities in scaling curves

## Key Papers

- **Hoffmann et al. (2022).** "Training Compute-Optimal Large Language Models" (Chinchilla). arXiv:2203.15556
- **Wei et al. (2022).** "Emergent Abilities of Large Language Models." arXiv:2206.07682
- **Xie et al. (2021).** "An Explanation of In-context Learning as Implicit Bayesian Inference." arXiv:2111.02080
- **Kaplan et al. (2020).** "Scaling Laws for Neural Language Models." arXiv:2001.08361

## Theoretical Background

### Power Law Scaling

Loss scales as a power law with compute, parameters, and data:

    L(C) = a * C^(-alpha) + L_inf

On a log-log plot, this is a straight line with slope -alpha and intercept at L_inf.

### Chinchilla-Optimal Ratio

Hoffmann et al. showed that parameters N and tokens D should scale equally:
N_opt ~ C^0.5, D_opt ~ C^0.5 (approximately). The FLOPs estimate is C ~ 6 N D.

### In-Context Learning as Bayesian Inference

Xie et al. propose that Transformer-based ICL implicitly performs Bayesian inference,
with the pretraining distribution serving as the prior and in-context examples as the
likelihood.

> **Math Callout:** Power laws and log-log plots, Bayesian inference (posterior = prior *
> likelihood / evidence), phase transitions in statistical physics.
> Resource: Kaplan et al. 2020 (arXiv:2001.08361).

## Exercises

1. `fit_power_law` -- Fit L(C) = a * C^(-alpha) + L_inf
2. `chinchilla_optimal_ratio` -- Compute optimal N and D given compute budget
3. `compute_flops_estimate` -- Approximate FLOPs as 6 * N * D
4. `bayesian_icl_posterior` -- In-context learning as Bayesian updating
5. `emergent_ability_metric` -- Detect phase transitions in scaling curves
