"""
Tests for Module 06: RNNs for Sequences

Run: pytest tests/test_exercises.py -v
"""

import numpy as np
import torch
import pytest

from exercises import (
    rnn_cell_manual,
    rnn_forward_manual,
    LSTMClassifier,
    tokenize_simple,
    train_sentiment_model,
)


# ---------------------------------------------------------------------------
# rnn_cell_manual
# ---------------------------------------------------------------------------

class TestRnnCellManual:
    def test_output_shape(self):
        h = rnn_cell_manual(
            x=np.zeros(3), h_prev=np.zeros(5),
            W_xh=np.random.randn(5, 3),
            W_hh=np.random.randn(5, 5),
            b_h=np.zeros(5),
        )
        assert h.shape == (5,)

    def test_tanh_range(self):
        """Output should be in [-1, 1] due to tanh."""
        np.random.seed(42)
        h = rnn_cell_manual(
            x=np.random.randn(3),
            h_prev=np.random.randn(5),
            W_xh=np.random.randn(5, 3),
            W_hh=np.random.randn(5, 5),
            b_h=np.random.randn(5),
        )
        assert np.all(h >= -1.0) and np.all(h <= 1.0)

    def test_zero_input_zero_hidden(self):
        """With zero inputs and zero bias, output should be tanh(0) = 0."""
        h = rnn_cell_manual(
            x=np.zeros(3), h_prev=np.zeros(5),
            W_xh=np.random.randn(5, 3),
            W_hh=np.random.randn(5, 5),
            b_h=np.zeros(5),
        )
        np.testing.assert_allclose(h, np.zeros(5), atol=1e-10)


# ---------------------------------------------------------------------------
# rnn_forward_manual
# ---------------------------------------------------------------------------

class TestRnnForwardManual:
    def test_output_shapes(self):
        np.random.seed(0)
        seq_len, input_size, hidden_size = 4, 3, 5
        X = np.random.randn(seq_len, input_size)
        h0 = np.zeros(hidden_size)
        W_xh = np.random.randn(hidden_size, input_size)
        W_hh = np.random.randn(hidden_size, hidden_size)
        b_h = np.zeros(hidden_size)

        all_hidden, h_final = rnn_forward_manual(X, h0, W_xh, W_hh, b_h)
        assert all_hidden.shape == (seq_len, hidden_size)
        assert h_final.shape == (hidden_size,)

    def test_final_matches_last_hidden(self):
        np.random.seed(0)
        X = np.random.randn(4, 3)
        h0 = np.zeros(5)
        W_xh = np.random.randn(5, 3)
        W_hh = np.random.randn(5, 5)
        b_h = np.zeros(5)

        all_hidden, h_final = rnn_forward_manual(X, h0, W_xh, W_hh, b_h)
        np.testing.assert_allclose(all_hidden[-1], h_final, atol=1e-10)

    def test_single_step_matches_cell(self):
        np.random.seed(0)
        X = np.random.randn(1, 3)
        h0 = np.random.randn(5)
        W_xh = np.random.randn(5, 3)
        W_hh = np.random.randn(5, 5)
        b_h = np.random.randn(5)

        all_hidden, h_final = rnn_forward_manual(X, h0, W_xh, W_hh, b_h)
        h_cell = rnn_cell_manual(X[0], h0, W_xh, W_hh, b_h)
        np.testing.assert_allclose(h_final, h_cell, atol=1e-10)


# ---------------------------------------------------------------------------
# LSTMClassifier
# ---------------------------------------------------------------------------

class TestLSTMClassifier:
    def test_output_shape(self):
        model = LSTMClassifier(vocab_size=100, embed_dim=16,
                               hidden_dim=32, num_classes=2)
        x = torch.randint(0, 100, (4, 10))  # batch=4, seq_len=10
        out = model(x)
        assert out.shape == (4, 2)

    def test_gradients_flow(self):
        model = LSTMClassifier(vocab_size=50, embed_dim=8,
                               hidden_dim=16, num_classes=2)
        x = torch.randint(0, 50, (2, 5))
        out = model(x)
        loss = out.sum()
        loss.backward()
        # Check that at least one parameter has a gradient
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad


# ---------------------------------------------------------------------------
# tokenize_simple
# ---------------------------------------------------------------------------

class TestTokenizeSimple:
    def test_output_shape(self):
        texts = ["hello world", "foo bar baz"]
        tokens, vocab = tokenize_simple(texts, max_vocab=100, max_len=5)
        assert tokens.shape == (2, 5)
        assert tokens.dtype == torch.long

    def test_padding(self):
        texts = ["hi"]
        tokens, vocab = tokenize_simple(texts, max_vocab=100, max_len=10)
        # "hi" is one token; rest should be PAD (0)
        assert tokens[0, 1:].sum().item() == 0  # all pad after first token

    def test_truncation(self):
        texts = ["a b c d e f g h"]
        tokens, _ = tokenize_simple(texts, max_vocab=100, max_len=3)
        assert tokens.shape == (1, 3)

    def test_vocab_has_special_tokens(self):
        texts = ["hello world"]
        _, vocab = tokenize_simple(texts, max_vocab=100, max_len=5)
        assert "<PAD>" in vocab
        assert "<UNK>" in vocab
        assert vocab["<PAD>"] == 0
        assert vocab["<UNK>"] == 1

    def test_unknown_words(self):
        texts = ["hello world"]
        tokens, vocab = tokenize_simple(texts, max_vocab=3, max_len=5)
        # With max_vocab=3 (PAD, UNK, + 1 word), at least one word maps to UNK
        unk_idx = vocab["<UNK>"]
        assert unk_idx in tokens[0].tolist()


# ---------------------------------------------------------------------------
# train_sentiment_model
# ---------------------------------------------------------------------------

class TestTrainSentimentModel:
    @pytest.mark.slow
    def test_overfits_small_data(self):
        texts = [
            "this movie is great", "wonderful film loved it",
            "amazing acting superb", "best movie ever seen",
            "terrible movie awful", "worst film boring bad",
            "horrible acting terrible", "bad movie do not watch",
        ]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        result = train_sentiment_model(
            texts, labels, epochs=50, lr=0.01, hidden_dim=32
        )
        assert result["train_accuracy"] > 0.8

    def test_returns_expected_keys(self):
        texts = ["good", "bad"]
        labels = [1, 0]
        result = train_sentiment_model(texts, labels, epochs=1)
        assert "model" in result
        assert "vocab" in result
        assert "train_losses" in result
        assert "train_accuracy" in result
