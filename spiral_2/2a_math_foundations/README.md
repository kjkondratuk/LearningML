# Phase 2A: Mathematical Foundations

> **Goal:** Build the mathematical toolkit that underpins all of ML. Every concept is
> implemented in NumPy. By the end of this phase you should be able to read matrix
> calculus notation in papers without flinching.

## Spiral 2 Principles

- **No scikit-learn in implementations.** Use only NumPy. Scikit-learn appears solely as
  a reference oracle in tests.
- **"Naive then Derive" pedagogy.** For every topic: (1) implement the brute-force
  version, (2) identify what is slow or mathematically unsound, (3) derive the
  efficient/correct version from first principles, (4) verify both produce the same
  results within numerical tolerance.
- **Paper-and-code derivations.** Each module README includes a "Derive It" section.
  Work the math on paper (or in LaTeX) before touching code.

## Module Index

| # | Module | Core Idea | Key Deliverable |
|---|--------|-----------|-----------------|
| 01 | [Linear Algebra](01_linear_algebra/) | Decompositions that power every ML algorithm | SVD image compression mini-project |
| 02 | [Calculus & Optimization](02_calculus_and_optimization/) | Gradients, Hessians, Taylor series | Gradient descent visualizer |
| 03 | [Probability & Statistics](03_probability_and_statistics/) | MLE, MAP, Bayesian updating | Bayesian coin-flip inference |
| 04 | [Optimization Methods](04_optimization_methods/) | SGD, Adam, L-BFGS and friends | Optimizer shootout |
| 05 | [Information Theory](05_information_theory/) | Entropy, KL divergence, mutual information | --- |
| 06 | [Capstone: Math Library](06_capstone_math_library/) | Package everything into a reusable `mathlib.py` | End-to-end linear regression using only mathlib |

## Estimated Timeline

4--6 weeks. Math is the foundation; do not rush.

## Prerequisites

- Spiral 1 completed (comfortable with NumPy, basic ML intuition)
- `numpy`, `matplotlib`, `scipy` (reference only), `scikit-learn` (oracle only)

## How to Work Through This Phase

1. Read the module README and its "Derive It" sections with pen and paper.
2. Watch / read the linked resources.
3. Implement the exercise stubs (they raise `NotImplementedError` until you fill them in).
4. Run the test suite -- tests are written first and tell you exactly what correct looks like.
5. Complete the mini-project if the module has one.
6. Move on only when all tests pass.
