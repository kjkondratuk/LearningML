# Module 05: Information Theory

> Cross-entropy loss is not an arbitrary choice -- it comes directly from
> information theory. This module builds the connection between entropy,
> KL divergence, and the loss functions used in classification.

## Learning Objectives

- Implement Shannon entropy, cross-entropy, and KL divergence.
- Understand why cross-entropy is the natural loss for classification.
- Understand KL divergence as the cost of using the wrong distribution.
- Derive the differential entropy of a Gaussian and the KL divergence
  between two Gaussians in closed form.

## Math Resources

| Resource | What to Focus On |
|----------|-----------------|
| Cover & Thomas, *Elements of Information Theory*, ch 1--2 | Rigorous foundations |
| Bishop, *PRML*, section 1.6 | Information theory in ML context |
| Murphy, *MLAPP*, ch 2.8 | Information theory review |
| Colah, "Visual Information Theory" (blog post) | Best visual explanation of entropy and KL |

## Derive It

1. **Entropy of a fair coin.** H(X) = -sum p(x) log p(x) = -2*(0.5 * log 0.5) = 1 bit.
   Now derive: uniform distribution over K items has entropy log_2(K).

2. **Cross-entropy as loss.** Show that minimizing cross-entropy H(p, q) w.r.t. q
   is equivalent to minimizing KL(p || q). Since H(p) is constant (data is fixed),
   the optimal q is p. This is why cross-entropy loss works.

3. **KL divergence is non-negative.** Prove KL(p||q) >= 0 using Jensen's inequality
   applied to the concavity of log.

4. **KL between two Gaussians.** Derive the closed form:
   KL(N(mu1, s1^2) || N(mu2, s2^2)) =
     log(s2/s1) + (s1^2 + (mu1-mu2)^2) / (2*s2^2) - 0.5

5. **Mutual information.** Derive I(X;Y) = H(X) + H(Y) - H(X,Y) = KL(p(x,y) || p(x)p(y)).

## "Naive then Derive" Challenge

Estimate entropy by generating samples, counting frequencies (plug-in estimator),
and computing entropy from the frequencies. Compare to the true entropy. Show the
bias of the plug-in estimator for small sample sizes -- it systematically
underestimates the true entropy.

## Exercises

See `exercises.py`.

## Style Notes

- Use natural logarithm (np.log) throughout, not log base 2 -- unless explicitly
  computing in "bits" (which will be noted in the docstring).
- Handle the p=0 edge case with the convention 0 * log(0) = 0.
