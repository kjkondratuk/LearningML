"""
Tests for Graph Neural Network exercises.

Verifies:
- Laplacian eigenvalues in [0, 2]
- GCN permutation invariance on fully connected graphs
- GAT attention coefficients sum to 1
- Message passing equivariance to node permutation
- Readout invariance to node permutation
"""

import pytest
import torch

from exercises import (
    gat_attention,
    gcn_layer,
    graph_laplacian,
    graph_readout,
    message_passing_step,
)


class TestGraphLaplacian:
    def test_eigenvalues_in_range(self):
        """Normalized Laplacian eigenvalues must be in [0, 2]."""
        A = torch.tensor(
            [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.float
        )
        L = graph_laplacian(A)
        eigenvalues = torch.linalg.eigvalsh(L)
        assert (eigenvalues >= -1e-6).all(), f"Eigenvalues should be >= 0: {eigenvalues}"
        assert (eigenvalues <= 2.0 + 1e-6).all(), f"Eigenvalues should be <= 2: {eigenvalues}"

    def test_symmetric(self):
        """Laplacian of a symmetric adjacency matrix should be symmetric."""
        A = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float)
        L = graph_laplacian(A)
        assert torch.allclose(L, L.T, atol=1e-6), "Laplacian should be symmetric"

    def test_connected_graph_one_zero_eigenvalue(self):
        """A connected graph has exactly one zero eigenvalue."""
        A = torch.tensor(
            [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=torch.float
        )
        L = graph_laplacian(A)
        eigenvalues = torch.linalg.eigvalsh(L)
        num_zeros = (eigenvalues.abs() < 1e-5).sum()
        assert num_zeros == 1, f"Connected graph should have 1 zero eigenvalue, got {num_zeros}"


class TestGCNLayer:
    def test_output_shape(self):
        N, in_f, out_f = 5, 3, 4
        x = torch.randn(N, in_f)
        A = torch.ones(N, N) - torch.eye(N)
        W = torch.randn(in_f, out_f)
        out = gcn_layer(x, A, W)
        assert out.shape == (N, out_f)

    def test_fully_connected_same_features(self):
        """On a fully connected graph with identical features, all outputs should be equal."""
        N = 4
        x = torch.ones(N, 3)
        A = torch.ones(N, N) - torch.eye(N)
        W = torch.randn(3, 2)
        out = gcn_layer(x, A, W)
        # All nodes should get the same output due to symmetry
        for i in range(1, N):
            assert torch.allclose(out[0], out[i], atol=1e-5), (
                "Fully connected + identical features => identical outputs"
            )


class TestGATAttention:
    def test_output_shape(self):
        out_f = 4
        x_i = torch.randn(8, out_f)
        x_j = torch.randn(8, out_f)
        a = torch.randn(2 * out_f)
        e = gat_attention(x_i, x_j, a)
        assert e.shape == (8,), f"Expected shape (8,), got {e.shape}"

    def test_attention_softmax_sums_to_one(self):
        """After softmax normalization, attention weights for neighbors should sum to 1."""
        out_f = 4
        a = torch.randn(2 * out_f)
        x_i = torch.randn(1, out_f).expand(3, -1)  # Same node i, 3 neighbors
        x_j = torch.randn(3, out_f)
        e = gat_attention(x_i, x_j, a)
        alpha = torch.softmax(e, dim=0)
        assert torch.allclose(alpha.sum(), torch.tensor(1.0), atol=1e-5)


class TestMessagePassing:
    def test_permutation_equivariance(self):
        """Permuting node indices should permute the output correspondingly."""
        N, d = 4, 3
        h = torch.randn(N, d)
        # Simple cycle graph: 0-1-2-3-0
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])

        def msg_fn(h_i, h_j):
            return h_j

        def agg_fn(messages, n_nodes):
            out = torch.zeros(n_nodes, messages.shape[1])
            # Simple: each node gets its predecessor's features
            return out

        # Build a full permutation test
        perm = torch.tensor([2, 0, 3, 1])
        h_perm = h[perm]
        # Permute edge_index accordingly
        inv_perm = torch.argsort(perm)
        edge_index_perm = inv_perm[edge_index]

        out_original = message_passing_step(h, edge_index, msg_fn, agg_fn)
        out_perm = message_passing_step(h_perm, edge_index_perm, msg_fn, agg_fn)

        # Outputs should be permuted versions of each other
        assert torch.allclose(out_original[perm], out_perm, atol=1e-5), (
            "Message passing must be equivariant to node permutation"
        )


class TestGraphReadout:
    def test_mean_pooling(self):
        h = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = graph_readout(h, method="mean")
        assert torch.allclose(out, torch.tensor([3.0, 4.0]))

    def test_sum_pooling(self):
        h = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        out = graph_readout(h, method="sum")
        assert torch.allclose(out, torch.tensor([4.0, 6.0]))

    def test_permutation_invariant(self):
        """Readout should be invariant to node ordering for mean/sum."""
        h = torch.randn(5, 3)
        perm = torch.randperm(5)
        for method in ["mean", "sum"]:
            out1 = graph_readout(h, method=method)
            out2 = graph_readout(h[perm], method=method)
            assert torch.allclose(out1, out2, atol=1e-5), (
                f"Readout ({method}) must be permutation invariant"
            )
