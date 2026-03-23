# 03 -- Probability & Statistics

## Learning Objectives

- Estimate probabilities from data
- Apply Bayes' theorem correctly (and understand the common mistake)
- Compute the Gaussian (normal) PDF
- Sample from distributions and verify with estimates
- Build a Naive Bayes classifier from scratch

## Why This Matters for ML

Machine learning is applied probability. Every classification model outputs
probabilities. Every loss function has a probabilistic interpretation.
Understanding Bayes' theorem and distributions gives you the language to
reason about uncertainty in predictions.

### Go Parallel

Probability in Go is straightforward -- it is just counting:

```go
func estimateProb(outcomes []bool) float64 {
    count := 0
    for _, o := range outcomes {
        if o { count++ }
    }
    return float64(count) / float64(len(outcomes))
}
```

Bayes' theorem is a formula: `P(A|B) = P(B|A) * P(A) / P(B)`. The hard part
is not the code -- it is remembering to use it when your intuition fails.

## Key Concepts

### 1. Estimating Probability

```
P(event) ~ count(event) / count(total)
```

The law of large numbers says this estimate converges to the true probability.

### 2. Bayes' Theorem

```
P(A|B) = P(B|A) * P(A) / P(B)
```

> **Math Callout:** This is the most important formula in ML. It tells you
> how to update your beliefs (prior `P(A)`) when you see evidence (likelihood
> `P(B|A)`). Logistic regression, Naive Bayes, and even neural network outputs
> all relate to Bayes' theorem.

### 3. The Disease Test Problem

A test is 99% accurate. The disease affects 1% of the population. You test
positive. What is the probability you have the disease?

Intuition says 99%. Bayes says ~50%. This is the base rate fallacy, and it
trips up everyone.

### 4. Gaussian PDF

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))
```

The bell curve. Most ML assumes Gaussian noise (linear regression, Naive
Bayes with continuous features).

### 5. Naive Bayes Classification

Assumes features are conditionally independent given the class label:
```
P(class | features) ~ P(class) * prod(P(feature_i | class))
```

Despite the naive independence assumption, it works surprisingly well for
text classification and spam filtering.

## "Wrong Way" Challenge: Confusing P(A|B) with P(B|A)

`P(positive test | disease)` = 0.99 (test accuracy).
`P(disease | positive test)` ~ 0.50 (what you actually want).

These are **not** the same. Confusing them is called the **prosecutor's
fallacy** and leads to bad decisions in medicine, law, and ML model evaluation.

## Style Notes

- Always use log-probabilities when multiplying many probabilities (avoids
  underflow).
- Test Bayes implementations with known analytical answers.

## Exercises

See [exercises.py](exercises.py) for 6 stubs. Tests in
`tests/test_exercises.py`.
