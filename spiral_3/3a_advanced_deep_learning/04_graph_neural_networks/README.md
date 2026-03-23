# Module 04: Graph Neural Networks

## Learning Objectives

- Understand the message passing framework as the unifying paradigm for GNNs
- Implement GCN and GAT layers from their respective papers
- Connect spectral graph theory to spatial graph convolutions
- Verify permutation equivariance/invariance properties

## Key Papers

- **Kipf & Welling (2016).** "Semi-Supervised Classification with Graph Convolutional Networks." arXiv:1609.02907
- **Velickovic et al. (2017).** "Graph Attention Networks." arXiv:1710.10903
- **Gilmer et al. (2017).** "Neural Message Passing for Quantum Chemistry." arXiv:1704.01212

## Theoretical Background

### Graph Laplacian

For a graph with adjacency matrix A and degree matrix D, the normalized Laplacian is:

    L = I - D^{-1/2} A D^{-1/2}

Its eigenvalues are in [0, 2], and the number of zero eigenvalues equals the number of
connected components.

### GCN Layer

Kipf & Welling simplify spectral graph convolutions to:

    H' = sigma(D_hat^{-1/2} A_hat D_hat^{-1/2} H W)

where A_hat = A + I (self-loops) and D_hat is its degree matrix.

### GAT Attention

Graph Attention Networks compute attention coefficients:

    e_ij = LeakyReLU(a^T [W h_i || W h_j])
    alpha_ij = softmax_j(e_ij)

This allows nodes to attend to their neighbors with learned importance weights.

### Message Passing Neural Networks

The MPNN framework (Gilmer et al.) unifies GNNs as:

    m_i = AGG({M(h_i, h_j, e_ij) : j in N(i)})
    h_i' = U(h_i, m_i)

> **Math Callout:** Spectral graph theory (Chung, "Spectral Graph Theory" ch. 1-2),
> graph Laplacian eigenvalues, message passing as matrix multiplication.
> Resource: Distill.pub "A Gentle Introduction to Graph Neural Networks."

## Exercises

1. `graph_laplacian` -- Normalized graph Laplacian
2. `gcn_layer` -- Single GCN layer
3. `gat_attention` -- GAT attention coefficients
4. `message_passing_step` -- Generic MPNN step
5. `graph_readout` -- Global pooling (mean, sum, attention)

## Mini-Project

Node classification on the Cora citation dataset using a 2-layer GCN.

## Style Notes

- Use sparse representations where possible for efficiency.
- Test on small graphs (< 20 nodes) for correctness, then scale up.
