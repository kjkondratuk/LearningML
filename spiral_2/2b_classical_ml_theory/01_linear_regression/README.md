# Module 01: Linear Regression -- From MLE to MAP to Bayes

> Linear regression is not just "fit a line." It is the gateway to understanding
> MLE, MAP, Bayesian inference, and the connection between regularization and priors.

## Learning Objectives

- Derive the normal equations from minimizing RSS.
- Show that MLE under Gaussian noise is equivalent to OLS.
- Show that MAP with a Gaussian prior is ridge regression.
- Implement full Bayesian linear regression with posterior uncertainty.
- Understand numerical stability: SVD vs normal equations vs gradient descent.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Bishop, *PRML*, sections 3.1, 3.3 | Linear basis functions, Bayesian regression |
| Murphy, *MLAPP*, ch 7 | Complete treatment of linear regression |
| StatQuest, "Linear Regression" series | Clear visual explanations |

## Derive It

1. **Normal equations.** Minimize ||Xw - y||^2 w.r.t. w. Set the gradient to zero:
   X^T(Xw - y) = 0, solve for w = (X^T X)^{-1} X^T y.

2. **MLE is OLS.** Write the Gaussian log-likelihood log p(y|X,w,sigma^2) and show
   that maximizing it is equivalent to minimizing sum of squared residuals.

3. **MAP is Ridge.** Add a Gaussian prior N(0, tau^{-1} I) on w. Show that the MAP
   estimate is w = (X^T X + tau I)^{-1} X^T y = ridge regression.

4. **Bayesian posterior.** Derive the posterior p(w|X,y) when both prior and
   likelihood are Gaussian. Show that the posterior is also Gaussian with:
   - Posterior precision = prior precision + data precision
   - Posterior mean = posterior precision^{-1} @ (prior precision @ prior mean + X^T y / sigma^2)

5. **Predictive distribution.** Derive p(y*|x*, X, y) by integrating out w.
   Show that the predictive variance = noise variance + epistemic uncertainty.

## "Naive then Derive" Challenge

Fit a polynomial of degree 15 to 20 data points. Watch it overfit wildly. Then
derive why adding a Gaussian prior on weights (MAP = Ridge) fixes the overfitting.
Show the regularization path as prior precision varies.

## Exercises

See `exercises.py`.

## Mini-Project: Bayesian Linear Regression with Uncertainty

Fit Bayesian regression, plot uncertainty bands, show sequential updating.
See `mini_project.py`.
