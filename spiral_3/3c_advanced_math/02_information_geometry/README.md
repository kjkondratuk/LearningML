# Module 02: Information Geometry

## Learning Objectives

- Compute Fisher information matrices for common distributions
- Implement natural gradient descent and understand its advantages
- Understand KL divergence as a second-order approximation using Fisher information
- Compute geodesic distances in statistical manifolds

## Key References

- **Amari, S.** "Information Geometry and Its Applications." Springer, 2016.
- **Martens, J. (2014).** "New Insights and Perspectives on the Natural Gradient Method." arXiv:1412.1193

## Theoretical Background

The Fisher information matrix is:

    F_{ij} = E[(d/d theta_i log p(x|theta)) (d/d theta_j log p(x|theta))]

For a Gaussian N(mu, sigma^2): F = [[1/sigma^2, 0], [0, 2/sigma^2]].

The natural gradient uses F^{-1} to account for the geometry of the parameter space:

    theta <- theta - lr * F^{-1} * grad

> **Math Callout:** Riemannian manifolds (do Carmo "Riemannian Geometry" ch. 1),
> Fisher metric, Cramer-Rao bound. Resource: Amari "Information Geometry" ch. 1-3.

## Exercises

1. `fisher_information_gaussian` -- Fisher information for N(mu, sigma^2)
2. `fisher_information_bernoulli` -- Fisher information for Bernoulli(p)
3. `natural_gradient_step` -- Natural gradient update
4. `kl_divergence_second_order` -- KL ~ (1/2) delta^T F delta
5. `geodesic_distance_gaussians` -- Fisher-Rao distance
