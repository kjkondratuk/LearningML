# Phase 2C: Deep Learning (Theory Pass)

> **Goal:** Understand deep learning from the math up. Derive backpropagation,
> implement a neural network framework, understand why modern architectures work.
> Build a Transformer from scratch.

## Spiral 2 Principles

- Derive everything before implementing.
- Build in NumPy first (Modules 01--05), then switch to PyTorch (Modules 06--08)
  when raw NumPy becomes too slow for practical training.
- No `nn.Module` shortcuts in the core implementations -- build the components
  (attention, normalization, etc.) from tensors and operations.

## Module Index

| # | Module | Core Idea | Key Deliverable |
|---|--------|-----------|-----------------|
| 01 | [Backpropagation](01_backpropagation/) | Chain rule on computational graphs | Autograd engine |
| 02 | [Activations & Init](02_activation_and_initialization/) | Vanishing gradients, Xavier/He | Activation statistics tracker |
| 03 | [Optimization for DL](03_optimization_for_deep_learning/) | LR schedules, AdamW, warmup | LR finder |
| 04 | [Regularization for DL](04_regularization_for_deep_learning/) | Dropout, batch/layer norm | Regularization ablation study |
| 05 | [CNNs](05_cnns/) | Convolution, im2col, pooling | CNN from scratch in NumPy |
| 06 | [RNNs & LSTMs](06_rnns_and_lstms/) | BPTT, vanishing gradients, gating | Character-level language model |
| 07 | [Attention & Transformers](07_attention_and_transformers/) | Self-attention, multi-head, positional encoding | Transformer from scratch |
| 08 | [Capstone: Transformer](08_capstone_transformer/) | Full GPT/BERT-style model | Text generation |

## Estimated Timeline

6--8 weeks. Seven theory modules plus the Transformer capstone.

## Prerequisites

- Phase 2A (calculus, optimization) and Phase 2B (ML algorithms)
- PyTorch installed (for Modules 06--08)
