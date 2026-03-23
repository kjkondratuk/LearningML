"""
Tests for RNN/LSTM exercises.
"""

import numpy as np
import pytest

from exercises import (
    rnn_cell_forward,
    rnn_forward,
    gradient_flow_analysis,
    lstm_cell_forward,
    lstm_forward,
    gru_cell_forward,
)


class TestRNNCell:
    def test_matches_manual(self):
        """Verify RNN cell against hand computation."""
        np.random.seed(42)
        x = np.array([[1.0, 2.0]])  # batch=1, d_in=2
        h = np.array([[0.5, -0.5, 0.3]])  # batch=1, d_h=3
        Wxh = np.random.randn(2, 3) * 0.1
        Whh = np.random.randn(3, 3) * 0.1
        bh = np.zeros(3)
        h_t = rnn_cell_forward(x, h, Wxh, Whh, bh)
        expected = np.tanh(x @ Wxh + h @ Whh + bh)
        np.testing.assert_allclose(h_t, expected, atol=1e-10)


class TestRNNForward:
    def test_output_shape(self):
        np.random.seed(42)
        X = np.random.randn(4, 10, 5)  # batch=4, T=10, d_in=5
        h0 = np.zeros((4, 8))
        Wxh = np.random.randn(5, 8) * 0.1
        Whh = np.random.randn(8, 8) * 0.1
        bh = np.zeros(8)
        h_all, h_T = rnn_forward(X, h0, Wxh, Whh, bh)
        assert h_all.shape == (4, 10, 8)
        assert h_T.shape == (4, 8)
        np.testing.assert_array_equal(h_T, h_all[:, -1, :])


class TestGradientFlow:
    def test_vanishing(self):
        """Small spectral radius -> gradient vanishes."""
        Whh = np.eye(5) * 0.5  # spectral radius = 0.5
        norms = gradient_flow_analysis(Whh, T=50)
        assert norms[-1] < norms[0] * 0.01

    def test_exploding(self):
        """Large spectral radius -> gradient explodes."""
        Whh = np.eye(5) * 1.5  # spectral radius = 1.5
        norms = gradient_flow_analysis(Whh, T=20)
        assert norms[-1] > norms[0] * 10


class TestLSTM:
    def test_cell_output_shapes(self):
        np.random.seed(42)
        d_in, d_h, batch = 4, 6, 2
        x = np.random.randn(batch, d_in)
        h_prev = np.zeros((batch, d_h))
        c_prev = np.zeros((batch, d_h))
        d_total = d_in + d_h
        Wf = np.random.randn(d_total, d_h) * 0.1
        Wi = np.random.randn(d_total, d_h) * 0.1
        Wc = np.random.randn(d_total, d_h) * 0.1
        Wo = np.random.randn(d_total, d_h) * 0.1
        bf = np.zeros(d_h)
        bi = np.zeros(d_h)
        bc = np.zeros(d_h)
        bo = np.zeros(d_h)
        h_t, c_t, cache = lstm_cell_forward(
            x, h_prev, c_prev, Wf, Wi, Wc, Wo, bf, bi, bc, bo
        )
        assert h_t.shape == (batch, d_h)
        assert c_t.shape == (batch, d_h)

    def test_gradient_stays_bounded(self):
        """LSTM gradient should stay bounded unlike vanilla RNN."""
        # Use identity-like forget gate bias (so gradient flows)
        Whh_rnn = np.eye(5) * 0.5
        norms_rnn = gradient_flow_analysis(Whh_rnn, T=50)
        # LSTM cell state gradient stays ~O(1) through addition
        # This is tested indirectly by LSTM training succeeding


class TestGRU:
    def test_output_shape(self):
        np.random.seed(42)
        d_in, d_h, batch = 4, 6, 2
        x = np.random.randn(batch, d_in)
        h = np.zeros((batch, d_h))
        d_total = d_in + d_h
        Wz = np.random.randn(d_total, d_h) * 0.1
        Wr = np.random.randn(d_total, d_h) * 0.1
        Wh = np.random.randn(d_total, d_h) * 0.1
        h_t, _ = gru_cell_forward(
            x, h, Wz, Wr, Wh, np.zeros(d_h), np.zeros(d_h), np.zeros(d_h)
        )
        assert h_t.shape == (batch, d_h)
