# Module 05: Measure-Theoretic Probability for ML

## Learning Objectives

- Understand why measure theory matters for ML (continuous densities, change of variables)
- Implement Monte Carlo integration and importance sampling
- Compute Radon-Nikodym derivatives (density ratios)
- Apply the change-of-variables formula for normalizing flows

## Key References

- **Pollard, D.** "A User's Guide to Measure Theoretic Probability." Cambridge, 2001. Ch. 1-3.
- **Durrett, R.** "Probability: Theory and Examples." Ch. 1.

## Theoretical Background

### The Radon-Nikodym Theorem

If P is absolutely continuous with respect to Q, there exists a density ratio dP/dQ such that
E_P[f] = E_Q[f * dP/dQ]. This is the foundation of importance sampling.

### Change of Variables

For an invertible transformation y = f(x):
    p_Y(y) = p_X(f^{-1}(y)) * |det(J_{f^{-1}}(y))|

This is the mathematical basis for normalizing flows.

> **Math Callout:** Sigma-algebras (Pollard ch. 1), Lebesgue integration (ch. 2),
> Radon-Nikodym theorem, change of variables for normalizing flows.

## Exercises

1. `lebesgue_integral_mc` -- Monte Carlo integration
2. `importance_sampling` -- IS with variance computation
3. `radon_nikodym_discrete` -- Density ratio for discrete distributions
4. `change_of_variables` -- For normalizing flows
5. `kl_divergence_mc` -- Monte Carlo KL estimation
