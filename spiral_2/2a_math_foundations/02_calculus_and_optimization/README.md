# Module 02: Calculus and Optimization Foundations

> Gradients point the way downhill. This module builds the calculus machinery that
> drives every learning algorithm -- from the humble gradient to the full Hessian.

## Learning Objectives

- Compute numerical gradients, Jacobians, and Hessians from first principles.
- Derive and verify Taylor approximations.
- Implement gradient descent and gradient descent with momentum.
- Understand convexity and constrained optimization via Lagrange multipliers.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| 3Blue1Brown, *Essence of Calculus* series | Intuition for derivatives and the chain rule |
| 3Blue1Brown, "Gradient descent, how neural networks learn" (NN series ch 2) | Visual intuition for gradient descent |
| Boyd & Vandenberghe, *Convex Optimization*, ch 1--2 (free PDF) | Convexity definitions and first-order conditions |
| Murphy, *MLAPP*, ch 8.1--8.3 | Optimization in the ML context |

## Derive It

1. **Central differences.** Derive that the central difference approximation
   f'(x) ~ [f(x+e) - f(x-e)] / (2e) has O(e^2) error, while the forward
   difference [f(x+e) - f(x)] / e has only O(e) error.

2. **Gradient of a quadratic.** For f(x) = 0.5 x^T A x + b^T x + c, derive
   that grad f = Ax + b. This is the most important gradient in all of ML.

3. **Hessian symmetry.** Prove that if f has continuous second partial derivatives,
   then the Hessian is symmetric: d^2f/dx_i dx_j = d^2f/dx_j dx_i.

4. **Gradient descent convergence.** For a quadratic f(x) = 0.5 x^T A x, show
   that GD with learning rate lr converges if lr < 2/lambda_max(A). Derive the
   optimal learning rate.

5. **Momentum as a low-pass filter.** Show that the momentum update
   v_{t+1} = mu * v_t + grad_f is an exponential moving average of gradients,
   and derive the effective learning rate in terms of mu and lr.

## "Naive then Derive" Challenge

Implement gradient descent with a fixed learning rate on a poorly-conditioned
quadratic (ratio of eigenvalues = 100). Watch it zigzag. Then derive why
momentum helps: it dampens oscillation along the high-curvature direction while
accelerating along the low-curvature direction. Implement momentum and compare.

## Exercises

See `exercises.py` for stubs with detailed docstrings.

## Mini-Project: Gradient Descent Visualizer

Create contour plots for three functions and overlay optimization trajectories.
See `mini_project.py`.

## Style Notes

- Functions that compute gradients should accept a callable `f` and a point `x`.
- Use `epsilon=1e-5` as default for numerical differentiation.
- Return full trajectories from optimizers (list of all x values visited).
