# Module 02: Logistic Regression -- The Generalized Linear Model

> Logistic regression is classification's answer to linear regression. The sigmoid
> is not an arbitrary choice -- it is the canonical link function for the Bernoulli
> distribution. This module derives it from first principles.

## Learning Objectives

- Derive sigmoid as the canonical link function for Bernoulli.
- Derive the gradient and Hessian of the log-likelihood.
- Implement Newton-Raphson (IRLS) and show it converges much faster than GD.
- Extend to multi-class via softmax and cross-entropy.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Bishop, *PRML*, sections 4.1--4.3 | Discriminative models, IRLS |
| Murphy, *MLAPP*, ch 8.3--8.4 | Logistic regression derivation |
| Andrew Ng, CS229 lecture notes on logistic regression (free online) | Practical treatment |

## Derive It

1. **Sigmoid from exponential family.** Show that if y ~ Bernoulli(p) and we model
   log(p/(1-p)) = w^T x (log-odds is linear), then p = sigmoid(w^T x).

2. **Gradient of log-likelihood.** Derive grad = X^T (y - sigma(Xw)). Note the
   elegance: same form as linear regression's gradient.

3. **Hessian is NSD.** Derive H = -X^T diag(sigma(1-sigma)) X. Show it is negative
   semi-definite, proving the log-likelihood is concave (no local optima).

4. **Newton-Raphson / IRLS.** Derive the update: w_{new} = w - H^{-1} grad.
   Show this is equivalent to iteratively reweighted least squares.

5. **Softmax as sigmoid generalization.** Show that for K=2, softmax reduces to sigmoid.

## "Naive then Derive" Challenge

Implement logistic regression with vanilla gradient descent. Then derive Newton-Raphson.
Compare: Newton converges in 5-10 iterations; GD takes hundreds.

## Exercises

See `exercises.py`.

## Mini-Project: Multi-class MNIST

Train multinomial logistic regression on MNIST digits 0-4 from scratch.
See `mini_project.py`.
