# Phase 3B: NLP & Language Models

## Overview

This phase covers the foundations and frontiers of natural language processing, from
classical word embeddings through modern large language model training, alignment, and
scaling. Each module implements algorithms from landmark NLP papers.

## Prerequisites

- Spiral 2 sequence modeling (RNNs, basic attention)
- Probability (cross-entropy, KL divergence, Bayesian inference)
- Linear algebra (matrix factorization, SVD)
- PyTorch training loops and autograd

## Module Index

| # | Module | Key Paper(s) | Core Concept |
|---|--------|-------------|--------------|
| 01 | Word Embeddings | Mikolov et al. 2013, Pennington et al. 2014 | Distributional semantics, Skip-gram, GloVe |
| 02 | Transformer Deep Dive | Vaswani et al. 2017, Su et al. 2021 | Positional encodings, attention efficiency |
| 03 | Pretraining Objectives | Devlin et al. 2018, Radford et al. 2019 | MLM, CLM, ELECTRA |
| 04 | Alignment: RLHF & DPO | Ouyang et al. 2022, Rafailov et al. 2023 | Reward modeling, preference optimization |
| 05 | Scaling Laws & ICL | Hoffmann et al. 2022, Wei et al. 2022 | Power laws, emergent abilities |
| 06 | Capstone: Fine-tune LM | -- | LoRA fine-tuning + DPO alignment |

## Suggested Order

Sequential. Module 04 benefits from familiarity with policy gradients (Phase 3D-RL
Module 03), but can be done independently.

## Testing Philosophy

Tests verify mathematical properties of loss functions, convergence of training
objectives, and correctness of information-theoretic quantities (perplexity, KL divergence).
