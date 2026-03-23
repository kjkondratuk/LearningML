# Phase 3C: Advanced Math & Theory

## Overview

This phase covers the mathematical foundations underlying modern machine learning at
research depth: statistical learning theory, information geometry, kernel methods, convex
optimization, and measure-theoretic probability. Unlike Phases 3A-3B where you implement
algorithms, here you implement theoretical results and verify them computationally.

## Prerequisites

- Linear algebra (eigendecomposition, SVD, positive definiteness)
- Calculus and analysis (convergence, continuity, differentiation under the integral)
- Probability and statistics (distributions, expectation, concentration)
- Spiral 2 optimization modules

## Module Index

| # | Module | Key Reference | Core Concept |
|---|--------|--------------|--------------|
| 01 | Statistical Learning Theory | Shalev-Shwartz & Ben-David | PAC learning, VC dimension, Rademacher complexity |
| 02 | Information Geometry | Amari 1998, Martens 2014 | Fisher information, natural gradient |
| 03 | Kernel Methods & RKHS | Scholkopf & Smola | Reproducing kernels, Mercer's theorem |
| 04 | Convex Optimization | Boyd & Vandenberghe | Duality, KKT, proximal methods |
| 05 | Measure Theory for ML | Pollard | Sigma-algebras, Lebesgue integral, Radon-Nikodym |
| 06 | Capstone: Theory Project | Neyshabur et al. 2017 | Generalization bounds for neural networks |

## Suggested Order

Can be done alongside Phases 3A-3B. Module 03 (kernels) provides background for Bayesian
ML in Phase 3D. Module 04 (optimization) connects to all applied phases.

## Testing Philosophy

Tests verify mathematical theorems computationally: bounds hold, convergence rates match
theory, metric properties are satisfied (triangle inequality, symmetry, non-negativity).
