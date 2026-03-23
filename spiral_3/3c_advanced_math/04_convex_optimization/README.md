# Module 04: Convex Optimization

## Learning Objectives

- Verify convexity of functions computationally
- Implement and check KKT conditions
- Build proximal gradient descent (ISTA) for composite objectives
- Implement ADMM and Fenchel conjugates

## Key References

- **Boyd & Vandenberghe.** "Convex Optimization." Cambridge University Press, 2004. Ch. 3, 5, 9.
- **Parikh & Boyd.** "Proximal Algorithms." Foundations and Trends in Optimization, 2014.

> **Math Callout:** Convex conjugates (Boyd ch. 3), duality gap, strong duality via
> Slater's condition (ch. 5), proximal operators.

## Exercises

1. `verify_convexity` -- Convexity check via Jensen's inequality
2. `lagrangian` -- Form the Lagrangian
3. `kkt_conditions_check` -- Verify all four KKT conditions
4. `proximal_operator_l1` -- Soft-thresholding
5. `proximal_gradient_descent` -- ISTA for composite objectives
6. `admm_step` -- One step of ADMM
7. `fenchel_conjugate_quadratic` -- Compute f* for quadratic f
