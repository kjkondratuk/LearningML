# Phase 3A: Advanced Deep Learning

## Overview

This phase covers the frontier of deep generative and representational models. You will
implement algorithms directly from landmark papers, moving beyond textbook exercises into
research-level reproduction. Each module references specific papers by arXiv ID; you are
expected to read the original papers alongside these materials.

## Prerequisites

- Spiral 2 deep learning modules (backpropagation, CNNs, optimization)
- Comfortable with PyTorch autograd, tensor operations, and training loops
- Probability and statistics (distributions, KL divergence, maximum likelihood)
- Linear algebra (eigendecomposition, SVD, matrix calculus)

## Module Index

| # | Module | Key Paper(s) | Core Concept |
|---|--------|-------------|--------------|
| 01 | Variational Autoencoders | Kingma & Welling 2013 | Latent variable models, ELBO |
| 02 | Generative Adversarial Networks | Goodfellow et al. 2014, Arjovsky et al. 2017 | Minimax game, Wasserstein distance |
| 03 | Diffusion Models | Ho et al. 2020, Song et al. 2020 | Forward/reverse diffusion, score matching |
| 04 | Graph Neural Networks | Kipf & Welling 2016, Velickovic et al. 2017 | Message passing, spectral methods |
| 05 | Self-Supervised Learning | Chen et al. 2020, He et al. 2021 | Contrastive learning, masked modeling |
| 06 | Capstone: Generative Model | Ho et al. 2020 | Full DDPM reproduction |

## Suggested Order

Work through modules 01-05 sequentially, then tackle the capstone. Modules 01-03 form a
progression through generative modeling paradigms. Module 04 is somewhat independent.
Module 05 bridges generative and discriminative ideas.

## Testing Philosophy

Tests verify mathematical properties: non-negativity of divergences, convergence of
training objectives, correct distributional behavior of samples, and gradient flow through
reparameterization tricks.
