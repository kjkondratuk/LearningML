"""
Graph Neural Network Exercises

Implement GNN components from:
    Kipf & Welling (2016), arXiv:1609.02907
    Velickovic et al. (2017), arXiv:1710.10903
    Gilmer et al. (2017), arXiv:1704.01212
"""

import torch
from typing import Callable


def graph_laplacian(adjacency_matrix: torch.Tensor) -> torch.Tensor:
    """Compute the normalized graph Laplacian.

    L = I - D^{-1/2} A D^{-1/2}

    Args:
        adjacency_matrix: Symmetric adjacency matrix, shape (N, N).

    Returns:
        Normalized Laplacian, shape (N, N).
    """
    raise NotImplementedError


def gcn_layer(
    x: torch.Tensor, adjacency: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Single GCN layer from Kipf & Welling 2016.

    H' = sigma(D_hat^{-1/2} A_hat D_hat^{-1/2} H W)

    where A_hat = A + I (add self-loops), D_hat is the degree matrix of A_hat,
    and sigma is ReLU.

    Args:
        x: Node features, shape (N, in_features).
        adjacency: Adjacency matrix (without self-loops), shape (N, N).
        weights: Learnable weight matrix, shape (in_features, out_features).

    Returns:
        Updated node features, shape (N, out_features).
    """
    raise NotImplementedError


def gat_attention(
    x_i: torch.Tensor, x_j: torch.Tensor, a_weights: torch.Tensor
) -> torch.Tensor:
    """Compute GAT attention coefficient e_ij.

    e_ij = LeakyReLU(a^T [W h_i || W h_j])

    Note: the caller is responsible for the W projection. This function
    takes already-projected features.

    Args:
        x_i: Projected features of node i, shape (batch_size, out_features).
        x_j: Projected features of node j, shape (batch_size, out_features).
        a_weights: Attention weight vector, shape (2 * out_features,).

    Returns:
        Unnormalized attention coefficients, shape (batch_size,).
    """
    raise NotImplementedError


def message_passing_step(
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    message_fn: Callable,
    aggregate_fn: Callable,
) -> torch.Tensor:
    """Generic Message Passing Neural Network step.

    For each node i:
        messages = {message_fn(h_i, h_j) for j in neighbors(i)}
        h_i' = aggregate_fn(messages)

    Args:
        node_features: Node features, shape (N, feature_dim).
        edge_index: Edge list, shape (2, E), where edge_index[0] are source
                    and edge_index[1] are target nodes.
        message_fn: Function(h_i, h_j) -> message, where h_i is target, h_j is source.
        aggregate_fn: Function(messages, num_nodes) -> aggregated, e.g., sum or mean.

    Returns:
        Updated node features, shape (N, feature_dim).
    """
    raise NotImplementedError


def graph_readout(
    node_embeddings: torch.Tensor,
    method: str = "mean",
) -> torch.Tensor:
    """Global graph-level pooling.

    Args:
        node_embeddings: Node embeddings, shape (N, feature_dim).
        method: "mean", "sum", or "max".

    Returns:
        Graph-level embedding, shape (feature_dim,).
    """
    raise NotImplementedError
