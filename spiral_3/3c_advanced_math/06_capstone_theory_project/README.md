# Module 06: Capstone -- Theory Project

## Project Overview

Derive and empirically validate a generalization bound for a specific neural network architecture.

## Options

Choose one:
- **PAC-Bayes bound** for a 2-layer ReLU network
- **Norm-based generalization bound** following Neyshabur et al. 2017 (arXiv:1706.08947)

## Requirements

1. Write out the complete derivation (in LaTeX in the README or separate PDF)
2. Implement the bound computation given a trained network
3. Train networks of varying width on MNIST and plot bound vs actual test error
4. Show the bound is vacuous (> 1) for large networks -- this is the current state of theory
5. Implement a tighter bound (compression-based or PAC-Bayes with learned prior)
6. 2-page report discussing the gap between theory and practice

## Key Reference

- Neyshabur et al. (2017). "Exploring Generalization in Deep Networks." arXiv:1706.08947
