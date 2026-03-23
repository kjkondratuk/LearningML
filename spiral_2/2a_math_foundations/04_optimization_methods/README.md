# Module 04: Optimization Methods

> Every ML model is trained by an optimizer. This module implements the full
> family from vanilla SGD to Adam to L-BFGS, and builds intuition for when
> each is appropriate.

## Learning Objectives

- Implement SGD, momentum, Nesterov, AdaGrad, RMSProp, Adam from scratch.
- Understand per-parameter adaptive learning rates (AdaGrad/RMSProp/Adam).
- Implement a second-order method (L-BFGS) and understand the computational
  trade-offs against first-order methods.
- Implement backtracking line search with the Armijo condition.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Ruder, "An overview of gradient descent optimization algorithms" (2016, arXiv) | Best single survey of all optimizers |
| Boyd & Vandenberghe, *Convex Optimization*, ch 9--10 | Line search and Newton methods |
| Goodfellow, *Deep Learning*, ch 8 | Optimization for training neural networks |

## Derive It

1. **Nesterov look-ahead.** Show that Nesterov evaluates the gradient at the
   "look-ahead" point x + mu*v, not the current point x. Derive why this
   anticipatory correction leads to faster convergence on quadratics.

2. **AdaGrad's accumulator.** Show that AdaGrad divides each parameter's
   learning rate by the sqrt of the sum of squared past gradients for that
   parameter. Derive why parameters with large gradients get smaller effective
   learning rates.

3. **Adam = Momentum + RMSProp + bias correction.** Write out the first and
   second moment estimates m_t and v_t. Show that without bias correction,
   the estimates are biased toward zero at the start. Derive the correction
   factors 1/(1-beta1^t) and 1/(1-beta2^t).

4. **L-BFGS two-loop recursion.** Derive the algorithm that computes H_k * g
   using only the last m (gradient, position) pairs, without ever forming
   the full Hessian approximation.

## "Naive then Derive" Challenge

Train linear regression on data with features at very different scales
(one feature in [0,1], another in [0,100000]). Use vanilla SGD -- watch it
zigzag. Then derive why Adam's per-parameter learning rate fixes this.

## Exercises

See `exercises.py`.

## Mini-Project: Optimizer Shootout

Compare all optimizers on 4 surfaces with convergence curves and trajectory
plots. See `mini_project.py`.

## Style Notes

- All optimizers accept a `batch_fn` for stochastic gradient estimates.
- Return full loss histories for plotting convergence curves.
- Use default hyperparameters from the original papers (Adam: beta1=0.9, beta2=0.999).
