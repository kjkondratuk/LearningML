# Module 03: Probability and Statistics

> ML is applied statistics. This module builds the probabilistic foundations:
> likelihood, Bayes' rule, conjugate priors, and the MLE/MAP distinction that
> underpins every model you will ever train.

## Learning Objectives

- Implement Gaussian PDFs (univariate and multivariate) from the formula.
- Derive MLE and MAP estimators for the Gaussian and show the connection to
  sample mean/variance and regularization.
- Implement Bayesian updating on a discrete grid and with conjugate priors.
- Implement rejection sampling and demonstrate the Central Limit Theorem.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| 3Blue1Brown, "Bayes theorem" and "But what is a probability distribution?" | Geometric intuition for Bayes |
| Bishop, *PRML*, ch 1.2 (Probability Theory), 2.1--2.4 (Distributions) | Rigorous probability for ML |
| Murphy, *MLAPP*, ch 2 (Probability), 3 (Generative Models), 4 (Gaussians) | MLE, MAP, Bayesian inference |

## Derive It

1. **Gaussian MLE.** Write the log-likelihood for N i.i.d. samples from
   N(mu, sigma^2). Take derivatives w.r.t. mu and sigma^2, set to zero,
   and show that mu_MLE = sample mean, sigma^2_MLE = (biased) sample variance.

2. **MAP with Gaussian prior.** Add a Gaussian prior N(mu_0, sigma_0^2) on mu.
   Show that the MAP estimate is a precision-weighted average of the prior mean
   and the MLE, where precision = 1/variance.

3. **Beta-Binomial conjugacy.** Given a Beta(alpha, beta) prior on the coin
   probability p and k successes in n trials, derive that the posterior is
   Beta(alpha + k, beta + n - k). No integrals needed -- just pattern matching
   on the kernel of the distribution.

4. **Entropy of the Gaussian.** Derive H[N(mu, sigma^2)] = 0.5 * ln(2 pi e sigma^2).
   This tells you the Gaussian is the maximum-entropy distribution for a given
   mean and variance.

## "Naive then Derive" Challenge

Estimate the MLE of a Gaussian by brute-force grid search over (mu, sigma) pairs,
evaluating the log-likelihood at each. Then derive the closed-form solution. Verify
both give the same answer. The grid search takes seconds; the analytical version is
instant.

## Exercises

See `exercises.py`.

## Mini-Project: Bayesian Coin-Flip Inference

Watch a prior evolve into a posterior as evidence arrives, one flip at a time.
See `mini_project.py`.

## Style Notes

- Probability distributions return log-probabilities where numerically appropriate.
- Use `np.log` instead of `np.log2` (natural log is standard in ML).
- All random sampling functions should accept a `seed` parameter for reproducibility.
