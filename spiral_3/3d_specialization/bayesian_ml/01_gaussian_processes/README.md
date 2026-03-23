# Module 01: Gaussian Processes

## Key References
- **Rasmussen & Williams.** "Gaussian Processes for Machine Learning." Ch. 2.

## Theoretical Background
A GP defines a distribution over functions: f ~ GP(m(x), k(x,x')).
The posterior given data is also a GP with updated mean and covariance.
The log marginal likelihood enables principled kernel selection.

> **Math Callout:** Multivariate Gaussians, conditioning, Cholesky decomposition,
> marginal likelihood optimization.

## Exercises
1. `gp_prior_sample` -- Sample functions from GP prior
2. `gp_posterior` -- Compute posterior mean and variance
3. `rbf_kernel_with_hyperparams` -- ARD RBF kernel
4. `log_marginal_likelihood` -- Model evidence
5. `gp_hyperparameter_optimization` -- Optimize kernel hyperparams
