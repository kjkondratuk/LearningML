# Module 01: Word Embeddings

## Learning Objectives

- Understand the distributional hypothesis and its mathematical formalization
- Implement Skip-gram with negative sampling from scratch
- Implement GloVe's weighted least squares objective
- Evaluate embeddings with analogy tests and similarity benchmarks

## Key Papers

- **Mikolov et al. (2013).** "Efficient Estimation of Word Representations in Vector Space." arXiv:1301.3781
- **Pennington et al. (2014).** "GloVe: Global Vectors for Word Representation." EMNLP 2014.
- **Levy & Goldberg (2014).** "Neural Word Embedding as Implicit Matrix Factorization." NeurIPS.

## Theoretical Background

### Skip-gram with Negative Sampling

The Skip-gram model maximizes:

    J = log sigma(v_c^T v_w) + sum_{k=1}^K E_{w_k ~ P_n(w)}[log sigma(-v_{w_k}^T v_c)]

where v_c is the center word embedding and v_w is the context word embedding.

### GloVe Objective

GloVe learns embeddings by factoring the log co-occurrence matrix:

    J = sum_{i,j} f(X_ij) (w_i^T w_j + b_i + b_j - log X_ij)^2

where f(x) is a weighting function that caps at x_max.

### Connection to Matrix Factorization

Levy & Goldberg showed that Skip-gram with negative sampling implicitly factorizes
the pointwise mutual information (PMI) matrix.

> **Math Callout:** Noise Contrastive Estimation (Gutmann & Hyvarinen 2010), SVD
> relation to GloVe (Levy & Goldberg 2014), distributional semantics.
> Resource: Stanford CS224n Lecture 1-2.

## Exercises

1. `skipgram_forward` -- Skip-gram forward pass with negative sampling
2. `negative_sampling_loss` -- NEG loss function
3. `cbow_forward` -- Continuous Bag of Words forward pass
4. `glove_loss` -- GloVe weighted least squares loss
5. `analogy_test` -- Solve a:b :: c:? with vector arithmetic

## Style Notes

- Use two separate embedding matrices (input and output) for Skip-gram.
- GloVe uses symmetric embeddings; final embedding = (W + W_tilde) / 2.
