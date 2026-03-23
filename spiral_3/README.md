# Spiral 3: Advanced Topics & Research Frontier

## Overview

Spiral 3 is research-level. Exercises shift from "implement this function" to "implement
this paper's algorithm." Tests verify mathematical properties -- convergence rates, bounds,
distributional properties, invariances -- not just input/output correctness. Each module
references specific papers by arXiv ID. You are expected to read original papers and
reproduce key results.

## Structure

| Phase | Focus | Modules | Est. Hours |
|-------|-------|---------|-----------|
| **3A** Advanced Deep Learning | Generative models, GNNs, SSL | 6 (5 + capstone) | 60-80 |
| **3B** NLP & Language Models | Embeddings through alignment | 6 (5 + capstone) | 60-80 |
| **3C** Advanced Math & Theory | Learning theory, optimization | 6 (5 + capstone) | 50-70 |
| **3D** Specialization Tracks | RL / CV / Systems / Bayesian | 24 (4 tracks) | 240-320 |
| **3E** Paper Reproduction | Read, reproduce, extend | 5 (3 papers + capstone) | 80-120 |
| **Total** | | **47 modules** | **490-670** |

## Prerequisites

- Completion of Spirals 1 and 2
- Comfortable with PyTorch, NumPy, and scientific Python
- Probability, statistics, linear algebra, calculus
- Experience training deep learning models

## Phase Dependencies

    3A  -----+
              \
    3B  ------+---> 3E (Paper Reproduction)
              /
    3C  -----+---> 3D (Specializations)

- **3A, 3B, 3C** can start independently and in parallel
- **3D** specialization tracks can be done in any order; some benefit from earlier phases
- **3E** should be done last -- it synthesizes everything

## Testing Philosophy

Tests at this level verify **mathematical properties**:
- **Convergence:** Loss decreases. Algorithms converge to known solutions.
- **Bounds:** Generalization bounds are valid. KL divergence is non-negative.
- **Invariances:** Permutation invariance/equivariance of GNNs. Translation equivariance.
- **Limits:** When a parameter goes to 0 or infinity, behavior matches theory.
- **Statistical:** Distributions match expected (KS test, chi-squared). Estimators are unbiased.

## How to Work Through Spiral 3

1. **Read the paper first.** Every module references specific papers. Read them.
2. **Derive before implementing.** Work through the math on paper before coding.
3. **Run tests early.** The tests define what "correct" means. Let them guide you.
4. **Compare to known results.** Your implementations should match published numbers
   (within compute budget limitations).
5. **Write it up.** Capstone projects include reports. Treat them as practice for
   writing research papers.

## Directory Structure

```
spiral_3/
├── 3a_advanced_deep_learning/     # VAEs, GANs, Diffusion, GNNs, SSL
├── 3b_nlp_language_models/        # Embeddings, Transformers, RLHF, Scaling
├── 3c_advanced_math/              # Learning theory, Info geometry, Kernels, Optimization
├── 3d_specialization/
│   ├── reinforcement_learning/    # MDPs through offline RL
│   ├── computer_vision/           # Detection through NeRFs
│   ├── ml_systems/                # Distributed training through serving
│   └── bayesian_ml/               # GPs through Bayesian optimization
└── 3e_paper_reproduction/         # Reading methodology, 3 paper reproductions, capstone
```
